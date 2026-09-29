"""
Downloads ERA5 hourly total_precipitation (all 24 hours/day) for one year,
India bounding box -- needed to fix Bug 2 (the 4-times-daily sample only
captured ~4/24 of each day's rain, since CDS delivers ERA5 tp as clean
1-hour increments, not 6-hour accumulations).

Usage: data/.venv/bin/python3 data/scripts/download_era5_hourly_tp.py 2015
"""
import sys
import os
import cdsapi

OUT_DIR = "data/raw/era5_hourly_tp"
AREA = [38, 68, 6, 98]  # N, W, S, E -- matches the original ERA5 pull's bbox


def download_year(year: int):
    out_path = os.path.join(OUT_DIR, f"era5_tp_hourly_{year}.nc")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1_000_000:
        print(f"{year}: already downloaded, skipping")
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    c = cdsapi.Client()
    print(f"{year}: submitting CDS request for 24-hourly total_precipitation...")
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": "total_precipitation",
            "year": str(year),
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": AREA,
            "format": "netcdf",
        },
        out_path,
    )
    print(f"{year}: wrote {out_path} ({os.path.getsize(out_path):,} bytes)")


if __name__ == "__main__":
    years = [int(y) for y in sys.argv[1:]]
    if not years:
        raise SystemExit("usage: download_era5_hourly_tp.py YEAR [YEAR ...]")
    for y in years:
        download_year(y)
