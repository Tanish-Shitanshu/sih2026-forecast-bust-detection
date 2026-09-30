"""Bulk download of NCMRWF's operational ensemble (NEPS, control member) from TIGGE on the ECMWF
Data Store, reduced to subdivision rows as it goes.

Needs an ECMWF Data Store API key in ~/.cdsapirc:
    url: https://ecds.ecmwf.int/api
    key: <your token from https://ecds.ecmwf.int/profile>
and the TIGGE licence accepted once on the dataset's Download page.

    python ml/scripts/fetch_tigge_ncmrwf.py --start 2017-08-01 --end 2025-12-31 --out <dir> [--workers 4]

Per MONTH it makes two requests (the web form allows one date, but the API accepts a list of
days; one month of rain = ~40 MB, ~3 min). ECMWF limits queued requests per user, so keep
--workers at 2:
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


def _req(month, days, variables, hours):
    return {"origin": "ncmrwf", "year": f"{month:%Y}", "month": f"{month:%m}", "day": [f"{d:%d}" for d in days],
            "time": "00:00", "level_type": "single_level", "variable": variables,
            "forecast_type": "control_forecast", "leadtime_hour": [str(h) for h in hours],
            "data_format": "grib", "area": AREA}


def fetch_month(client, month, days, tmp, weights, retries=4):
    """Download one month (rain + issue-time state), reduce to subdivisions, split per date."""
    parts = []
    for name, vars_, hours in (("rain", ["total_precipitation"], range(0, 241, 24)), ("state", STATE_VARS, [0])):
        f = tmp / f"{month:%Y%m}_{name}.grib"
        if not (f.exists() and f.stat().st_size > 1000):
            for attempt in range(retries):
                try:
                    client.retrieve("tigge-forecasts", _req(month, days, vars_, hours)).download(str(f))
                    break
                except Exception:
                    if attempt == retries - 1:
                        raise
                    time.sleep(120 * (attempt + 1))
        with _DECODE:
            parts.append(to_subdivisions(read_request(f), weights))
        f.unlink(missing_ok=True)
    out = parts[0].merge(parts[1], on=["date", "subdivision_code", "lead_day"], how="outer")
    return {d: g for d, g in out.groupby("date")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2017-08-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    import cdsapi

    out = Path(a.out)
    (out / "parts").mkdir(parents=True, exist_ok=True)
    tmp = out / "tmp"
    tmp.mkdir(exist_ok=True)
    weights = pd.read_csv(WEIGHTS)
    days = [d for d in pd.date_range(a.start, a.end) if not (out / "parts" / f"{d:%Y%m%d}.parquet").exists()]
    months = {}
    for d in days:
        months.setdefault(d.to_period("M").to_timestamp(), []).append(d)
    print(f"{len(days)} dates in {len(months)} months to fetch ({a.start} .. {a.end}), {a.workers} parallel", flush=True)
    client = cdsapi.Client(quiet=True, progress=False)
    failed, done = [], 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(fetch_month, client, m, ds, tmp, weights): m for m, ds in months.items()}
        for fu in as_completed(futs):
            m = futs[fu]
            try:
                for d, g in fu.result().items():
                    g.to_parquet(out / "parts" / f"{pd.Timestamp(d):%Y%m%d}.parquet", index=False)
                done += 1
                print(f"  {done}/{len(months)} months done (last {m:%Y-%m})", flush=True)
            except Exception as e:  # e.g. a month missing from the archive
                failed.append((f"{m:%Y-%m}", str(e)[:160]))
    print(f"finished: {done} ok, {len(failed)} failed", flush=True)
    for d, e in failed[:50]:
        print("  FAILED", d, e)


if __name__ == "__main__":
    main()
