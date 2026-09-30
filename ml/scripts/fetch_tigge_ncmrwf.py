"""Bulk download of NCMRWF's operational ensemble (NEPS, control member) from TIGGE on the ECMWF
Data Store, reduced to subdivision rows as it goes.

Needs an ECMWF Data Store API key in ~/.cdsapirc:
    url: https://ecds.ecmwf.int/api
    key: <your token from https://ecds.ecmwf.int/profile>
and the TIGGE licence accepted once on the dataset's Download page.

    python ml/scripts/fetch_tigge_ncmrwf.py --start 2017-08-01 --end 2025-12-31 --out <dir> [--workers 4]

Per issue date (00Z) it makes two requests (the web form allows one date per request):
  rain : total precipitation, lead hours 0, 24, ..., 240   -> daily rain for lead days 1..10
  state: MSLP, surface pressure, 2 m temperature and dewpoint, 10 m winds at hour 0 (issue time)
Each GRIB is converted with vishwas_ml.ncmrwf_tigge.read_request + to_subdivisions (weights for
the NEPS grid in ml/data/weights_ncmrwf_neps.csv), written to <out>/parts/<YYYYMMDD>.parquet and
deleted. Re-running skips dates already done; failed dates are listed at the end.
"""
import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML))
from vishwas_ml.ncmrwf_s2s import to_subdivisions  # noqa: E402
from vishwas_ml.ncmrwf_tigge import read_request  # noqa: E402

WEIGHTS = ML / "data" / "weights_ncmrwf_neps.csv"
AREA = [38, 68, 6, 98]  # N, W, S, E
_DECODE = threading.Lock()  # eccodes/cfgrib are not thread-safe: downloads run in parallel, decoding does not
STATE_VARS = ["mean_sea_level_pressure", "surface_pressure", "2_m_temperature", "2_m_dewpoint_temperature",
              "10_m_u_component_of_wind", "10_m_v_component_of_wind"]


def _req(day, variables, hours):
    return {"origin": "ncmrwf", "year": f"{day:%Y}", "month": f"{day:%m}", "day": f"{day:%d}", "time": "00:00",
            "level_type": "single_level", "variable": variables, "forecast_type": "control_forecast",
            "leadtime_hour": [str(h) for h in hours], "data_format": "grib", "area": AREA}


def fetch_one(client, day, tmp, weights, retries=3):
    parts = []
    for name, vars_, hours in (("rain", ["total_precipitation"], range(0, 241, 24)), ("state", STATE_VARS, [0])):
        f = tmp / f"{day:%Y%m%d}_{name}.grib"
        for attempt in range(retries if not (f.exists() and f.stat().st_size > 1000) else 0):
            try:
                client.retrieve("tigge-forecasts", _req(day, vars_, hours)).download(str(f))
                break
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(30 * (attempt + 1))
        with _DECODE:
            parts.append(to_subdivisions(read_request(f), weights))
        f.unlink(missing_ok=True)
        for idx in tmp.glob(f"{f.name}*.idx"):
            idx.unlink(missing_ok=True)
    out = parts[0].merge(parts[1], on=["date", "subdivision_code", "lead_day"], how="outer")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2017-08-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    import cdsapi

    out = Path(a.out)
    (out / "parts").mkdir(parents=True, exist_ok=True)
    tmp = out / "tmp"
    tmp.mkdir(exist_ok=True)
    weights = pd.read_csv(WEIGHTS)
    days = [d for d in pd.date_range(a.start, a.end) if not (out / "parts" / f"{d:%Y%m%d}.parquet").exists()]
    print(f"{len(days)} dates to fetch ({a.start} .. {a.end}), {a.workers} parallel", flush=True)
    client = cdsapi.Client(quiet=True, progress=False)
    failed, done = [], 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(fetch_one, client, d, tmp, weights): d for d in days}
        for fu in as_completed(futs):
            d = futs[fu]
            try:
                fu.result().to_parquet(out / "parts" / f"{d:%Y%m%d}.parquet", index=False)
                done += 1
                if done % 25 == 0:
                    print(f"  {done}/{len(days)} done (last {d:%Y-%m-%d})", flush=True)
            except Exception as e:  # e.g. a date missing from the archive
                failed.append((f"{d:%Y-%m-%d}", str(e)[:120]))
    print(f"finished: {done} ok, {len(failed)} failed", flush=True)
    for d, e in failed[:50]:
        print("  FAILED", d, e)


if __name__ == "__main__":
    main()
