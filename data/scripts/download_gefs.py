"""
Downloads GEFSv12 reforecast total-precipitation (apcp_sfc) GRIB2 files for
the control member (c00), one file per forecast init date, from the public
NOAA S3 bucket (no login). Each file holds all Day 1-10 lead times for that
init date, globally, at 0.25 deg.

Usage: data/.venv/bin/python3 data/scripts/download_gefs.py 2015-06-01 2015-09-30
       (first two args: inclusive start/end date, YYYY-MM-DD; downloads every day in between)
"""
import sys
import os
import time
import datetime
import requests

BASE = "https://noaa-gefs-retrospective.s3.amazonaws.com/GEFSv12/reforecast"
OUT_DIR = "data/raw/gefs"
MAX_ATTEMPTS = 4
BACKOFF_SECONDS = [5, 15, 30]  # between attempts 1->2, 2->3, 3->4


def url_for(date: datetime.date) -> str:
    ymd = date.strftime("%Y%m%d")
    return f"{BASE}/{date.year}/{ymd}00/c00/Days%3A1-10/apcp_sfc_{ymd}00_c00.grib2"


def download_date(date: datetime.date):
    """
    Returns True (downloaded or already present), False (server says no
    reforecast for this date -- a real "skip", not an error). Raises only
    after MAX_ATTEMPTS transient network failures, so a persistent problem
    still surfaces rather than silently producing a gappy dataset.
    """
    ymd = date.strftime("%Y%m%d")
    out_path = os.path.join(OUT_DIR, f"apcp_sfc_{ymd}00_c00.grib2")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1_000_000:
        print(f"{ymd}: already downloaded, skipping")
        return True
    url = url_for(date)

    last_exc = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = requests.get(url, timeout=180)
            if resp.status_code != 200:
                print(f"{ymd}: HTTP {resp.status_code}, skipping (no reforecast for this date)")
                return False
            with open(out_path, "wb") as fh:
                fh.write(resp.content)
            print(f"{ymd}: wrote {out_path} ({len(resp.content):,} bytes)")
            return True
        except requests.exceptions.RequestException as e:
            last_exc = e
            if attempt < MAX_ATTEMPTS - 1:
                wait = BACKOFF_SECONDS[attempt]
                print(f"{ymd}: transient network error ({e}), retrying in {wait}s "
                      f"(attempt {attempt + 1}/{MAX_ATTEMPTS})")
                time.sleep(wait)
    raise RuntimeError(f"{ymd}: download failed after {MAX_ATTEMPTS} attempts") from last_exc


def daterange(start: datetime.date, end: datetime.date):
    d = start
    while d <= end:
        yield d
        d += datetime.timedelta(days=1)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    if len(sys.argv) != 3:
        raise SystemExit("usage: download_gefs.py START_DATE END_DATE  (YYYY-MM-DD, inclusive)")
    start = datetime.date.fromisoformat(sys.argv[1])
    end = datetime.date.fromisoformat(sys.argv[2])
    ok, fail = 0, 0
    for d in daterange(start, end):
        if download_date(d):
            ok += 1
        else:
            fail += 1
    print(f"\nDone: {ok} downloaded/present, {fail} failed/missing")
