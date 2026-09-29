"""Grid cell -> IMD subdivision area weights, for any regular lat/lon grid.

Boundaries: IMD's meteorological subdivision shapefile, `indian_met_zones.v2.*` from
github.com/India-Meteorological-Department/Indian_met_zones (Unlicense, 36 subdivisions, WGS84).
The frontend lists 33: Andaman & Nicobar, Lakshadweep and Uttarakhand are not in it and are dropped.

weight = (area of the cell inside the subdivision, in degrees^2) x cos(latitude), so a
subdivision mean is an area-weighted mean over every cell that touches it, partial cells
counting by their overlap. Pass the result to ncmrwf_s2s.to_subdivisions().
"""
import numpy as np
import pandas as pd

# Shapefile ST_NM -> frontend subdivision code (ml/data/subdivisions.json).
SHAPE_TO_CODE = {
    "NMMT": "NE.HL", "Arunachal Pradesh": "AR", "Chhattisgarh": "CG", "Jammu & Kashmir": "J&K",
    "Vidarbha": "VID", "Marathwada": "MWD", "Madhya Maharashtra": "MDH.MH", "West Rajasthan": "W.RJ",
    "Telangana": "TG", "Rayalaseema": "RYL", "S.I. Karnataka": "SI.KA", "Kerala": "KL",
    "Tamil Nadu & Puducherry": "TN/PY", "Konkan & Goa": "KNK/GA", "Coastal Andhra Pradesh": "CST.AP",
    "Gujarat region": "GJ", "Saurashtra & Kachh": "SAU/KCH", "Coastal Karnataka": "CST.KA",
    "N.I. Karnataka": "NI.KA", "East Rajasthan": "E.RJ", "West Madhya Pradesh": "W.MP",
    "East Madhya Pradesh": "E.MP", "Haryana, CHD & Delhi": "HR/DL", "Punjab": "PB", "Himachal Pradesh": "HP",
    "East Uttar Pradesh": "E.UP", "West Uttar Pradesh": "W.UP", "Assam & Meghalaya": "ASM",
    "Gangetic West Bengal": "GWB", "Odisha": "OD", "Jharkhand": "JH", "Bihar": "BR", "SHWB": "SIK/NWB",
}
NOT_IN_FRONTEND = {"Andaman & Nicobar Island", "Lakshadweep", "Uttarakhand"}


def load_subdivisions(shp_base, simplify=0.01):
    """{code: shapely geometry} for the 33 frontend subdivisions. shp_base: path without the
    .shp/.shx/.dbf extension (e.g. .../indian_met_zones.v2)."""
    import shapefile
    from shapely.geometry import shape

    b = str(shp_base)
    with open(b + ".shp", "rb") as shp, open(b + ".shx", "rb") as shx, open(b + ".dbf", "rb") as dbf:
        r = shapefile.Reader(shp=shp, shx=shx, dbf=dbf)
        out = {}
        for sr in r.iterShapeRecords():
            name = sr.record["ST_NM"]
            if name in NOT_IN_FRONTEND:
                continue
            if name not in SHAPE_TO_CODE:
                raise KeyError(f"unmapped subdivision in shapefile: {name!r}")
            g = shape(sr.shape.__geo_interface__).buffer(0)
            out[SHAPE_TO_CODE[name]] = g.simplify(simplify, preserve_topology=True) if simplify else g
    missing = set(SHAPE_TO_CODE.values()) - set(out)
    if missing:
        raise ValueError(f"subdivisions missing from shapefile: {sorted(missing)}")
    return out


def build_weights(lats, lons, subdivisions, dlat=None, dlon=None):
    """Weights for the regular grid with cell centres lats x lons (1-D, degrees east -180..180).
    Returns latitude, longitude, subdivision_code, weight, frac (share of the cell inside)."""
    from shapely import STRtree, box

    lats, lons = np.sort(np.unique(lats)), np.sort(np.unique(lons))
    dlat = dlat or float(np.median(np.diff(lats)))
    dlon = dlon or float(np.median(np.diff(lons)))
    codes = list(subdivisions)
    geoms = [subdivisions[c] for c in codes]
    tree = STRtree(geoms)
    rows = []
    for la in lats:
        coslat = np.cos(np.radians(la))
        for lo in lons:
            cell = box(lo - dlon / 2, la - dlat / 2, lo + dlon / 2, la + dlat / 2)
            for j in tree.query(cell, predicate="intersects"):
                a = cell.intersection(geoms[j]).area
                if a > 0:
                    rows.append((float(la), float(lo), codes[j], a * coslat, a / cell.area))
    return pd.DataFrame(rows, columns=["latitude", "longitude", "subdivision_code", "weight", "frac"])
