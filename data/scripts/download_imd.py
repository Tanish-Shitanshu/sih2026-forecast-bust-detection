"""
Downloads IMD Gridded Rainfall (0.25 deg, daily) yearly NetCDF files.
No login required — a plain POST to RF25.php with the year.

Usage: data/.venv/bin/python3 data/scripts/download_imd.py 2015 2016 2017 2018 2019
"""
import sys
import os
import requests

URL = "https://imdpune.gov.in/cmpg/Griddata/RF25.php"
OUT_DIR = "data/raw/imd"


def download_year(year: int):
    out_path = os.path.join(OUT_DIR, f"ind{year}_rfp25.nc")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1_000_000:
        print(f"{year}: already downloaded ({os.path.getsize(out_path):,} bytes), skipping")
        return
    print(f"{year}: downloading...")
    resp = requests.post(URL, data={"RF25": str(year)}, timeout=180)
    resp.raise_for_status()
    if len(resp.content) < 1_000_000:
        raise RuntimeError(f"{year}: response too small ({len(resp.content)} bytes) — likely an error page, not a NetCDF")
    with open(out_path, "wb") as fh:
        fh.write(resp.content)
    print(f"{year}: wrote {out_path} ({len(resp.content):,} bytes)")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    years = [int(y) for y in sys.argv[1:]]
    if not years:
        raise SystemExit("usage: download_imd.py YEAR [YEAR ...]")
    for y in years:
        download_year(y)
