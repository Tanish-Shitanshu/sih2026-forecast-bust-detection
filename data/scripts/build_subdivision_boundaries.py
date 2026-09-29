"""
Maps the official IMD 36-subdivision boundary file (mausam.imd.gov.in) down
to our 33 subdivisions (36 minus Lakshadweep, A & N Islands, Uttarakhand —
none of which appear in the frontend's subdivision list) and relabels each
feature with our subdivision_id/code/name from subdivisions.csv, so every
downstream script can join on subdivision_code without re-deriving this map.

Source of RAW_PATH: the IMD file ships with no CRS and unlabeled projected
coordinates (not lon/lat). Rather than guess the projection, we use a
community-corrected WGS84 re-projection of the same official dataset
(github.com/planemad, gist 1604109e8057cb6d6822e8909468f16c), verified here
by: (a) all 36 "subdivisio" names matching the raw file exactly, (b) a
CRS84 declaration, (c) the overall bounding box landing at lon 68-98 /
lat 6-38 as expected for India, and (d) centroids of small/compact
subdivisions (Kerala, Tamil Nadu) matching known geography within ~15km.
"""
import json
import csv

RAW_PATH = "data/raw/imd/subdivision_boundaries_raw.geojson"
SUBDIVISIONS_CSV = "data/scripts/subdivisions.csv"
OUT_PATH = "data/processed/subdivision_boundaries.geojson"

# raw "subdivisio" name -> our subdivision_code (frontend/index.html SUBS[].code)
RAW_NAME_TO_CODE = {
    "COASTAL ANDHRA PRADESH": "CST.AP",
    "ARUNACHAL PRADESH": "AR",
    "ASSAM & MEGHALAYA": "ASM",
    "BIHAR": "BR",
    "KONKAN & GOA": "KNK/GA",
    "SAURASHTRA & KUTCH": "SAU/KCH",
    "HAR. CHD & DELHI": "HR/DL",
    "HIMACHAL PRADESH": "HP",
    "JAMMU & KASHMIR": "J&K",
    "S. I. KARNATAKA": "SI.KA",
    "KERALA": "KL",
    "WEST MADHYA PRADESH": "W.MP",
    "MADHYA MAHARASHTRA": "MDH.MH",
    "N M M T": "NE.HL",
    "ORISSA": "OD",
    "PUNJAB": "PB",
    "WEST RAJASTHAN": "W.RJ",
    "TAMILNADU & PONDICHERY": "TN/PY",
    "EAST UTTAR PRADESH": "E.UP",
    "GANGETIC WEST BENGAL": "GWB",
    "SHWB & SIKKIM": "SIK/NWB",
    "JHARKHAND": "JH",
    "EAST RAJASTHAN": "E.RJ",
    "GUJARAT REGION": "GJ",
    "WEST UTTAR PRADESH": "W.UP",
    "TELANGANA": "TG",
    "RAYALASEEMA": "RYL",
    "CHHATTISGARH": "CG",
    "EAST MADHYA PRADESH": "E.MP",
    "COASTAL KARNATAKA": "CST.KA",
    "N. I. KARNATAKA": "NI.KA",
    "VIDARBHA": "VID",
    "MARATHWADA": "MWD",
}
# Present in the raw 36 but not one of our 33 — dropped intentionally.
EXCLUDED_RAW_NAMES = {"LAKSHADWEEP", "A & N ISLAND", "UTTARAKHAND"}


def main():
    raw = json.load(open(RAW_PATH))

    subdivisions = {}
    with open(SUBDIVISIONS_CSV) as fh:
        for row in csv.DictReader(fh):
            subdivisions[row["subdivision_code"]] = row

    raw_names_seen = {f["properties"]["subdivisio"] for f in raw["features"]}
    expected_raw_names = set(RAW_NAME_TO_CODE) | EXCLUDED_RAW_NAMES
    if raw_names_seen != expected_raw_names:
        missing = expected_raw_names - raw_names_seen
        extra = raw_names_seen - expected_raw_names
        raise SystemExit(f"Raw name set mismatch. Missing: {missing}  Unexpected: {extra}")

    out_features = []
    codes_used = set()
    for f in raw["features"]:
        raw_name = f["properties"]["subdivisio"]
        if raw_name in EXCLUDED_RAW_NAMES:
            continue
        code = RAW_NAME_TO_CODE[raw_name]
        sub = subdivisions[code]
        codes_used.add(code)
        out_features.append({
            "type": "Feature",
            "properties": {
                "subdivision_id": int(sub["subdivision_id"]),
                "subdivision_code": code,
                "subdivision_name": sub["subdivision_name"],
            },
            "geometry": f["geometry"],
        })

    if codes_used != set(subdivisions):
        raise SystemExit(f"Code coverage mismatch. Missing from boundaries: {set(subdivisions) - codes_used}")
    if len(out_features) != 33:
        raise SystemExit(f"Expected 33 output features, got {len(out_features)}")

    out = {"type": "FeatureCollection", "features": out_features}
    with open(OUT_PATH, "w") as fh:
        json.dump(out, fh)
    print(f"Wrote {OUT_PATH}: {len(out_features)} subdivisions, all 33 codes matched exactly once.")


if __name__ == "__main__":
    main()
