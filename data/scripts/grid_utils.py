"""Shared helpers for assigning 0.25-degree grid cells to IMD subdivisions."""
import json
import numpy as np
from shapely.geometry import shape, Point
from shapely.strtree import STRtree

BOUNDARIES_PATH = "data/processed/subdivision_boundaries.geojson"


def load_boundaries():
    """Returns (geoms: list[shapely geometry], codes: list[str]) for the 33 subdivisions."""
    d = json.load(open(BOUNDARIES_PATH))
    geoms, codes = [], []
    for f in d["features"]:
        geoms.append(shape(f["geometry"]))
        codes.append(f["properties"]["subdivision_code"])
    return geoms, codes


def build_grid_subdivision_map(lats, lons, geoms=None, codes=None):
    """
    For a regular lat/lon grid (1-D lats, 1-D lons in degrees, -180..180 or
    0..360 — normalized internally), returns a dict mapping
    (lat_index, lon_index) -> subdivision_code for every grid-cell CENTER
    that falls inside one of the 33 subdivision polygons. Cells outside all
    subdivisions (ocean, other countries) are simply absent from the dict.

    Uses an STRtree for efficient point-in-polygon lookup, and a bounding-box
    pre-filter on lon so this is cheap even for a global GEFS grid.
    """
    if geoms is None or codes is None:
        geoms, codes = load_boundaries()

    tree = STRtree(geoms)
    geom_to_code = {id(g): c for g, c in zip(geoms, codes)}

    lons_norm = np.where(lons > 180, lons - 360, lons)
    # India bounding box, generous margin
    lon_mask = (lons_norm >= 60) & (lons_norm <= 100)
    lat_mask = (lats >= 5) & (lats <= 40)

    mapping = {}
    for i, lat in enumerate(lats):
        if not lat_mask[i]:
            continue
        for j, lon in enumerate(lons):
            if not lon_mask[j]:
                continue
            pt = Point(lons_norm[j], lat)
            candidate_idx = tree.query(pt)
            for idx in candidate_idx:
                g = geoms[idx]
                if g.contains(pt):
                    mapping[(i, j)] = codes[idx]
                    break
    return mapping
