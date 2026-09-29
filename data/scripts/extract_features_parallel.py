"""
Parallel version of extract_features.py: a process pool (not threads --
cfgrib/eccodes has real thread-safety caveats for concurrent C-library
calls within one process) where each worker downloads+decodes+aggregates
one date and writes its OWN atomic per-date part file
(data/processed/_parts/{year}/{date_str}.parquet). No process ever reads,
modifies, and rewrites a file another process might also be writing --
that pattern silently loses rows under concurrency (last writer wins).
A separate merge step concatenates all part files into the final
_features_{year}.parquet, also written atomically (temp file + os.replace).

Reference data (IMD daily rainfall incl. next-year spillover, ERA5 daily
features, the GEFS grid map) is loaded ONCE in this parent process and
handed to every worker via the Pool initializer, rather than reloaded
redundantly by each of N workers -- avoids both wasted work and an N-way
memory spike from each worker separately parsing the ~265MB ERA5 file.

Worker count is chosen from currently available RAM (psutil), not
hardcoded, budgeting conservatively per worker's peak GEFS-decode memory.

Usage:
  data/.venv/bin/python3 data/scripts/extract_features_parallel.py 2016
  data/.venv/bin/python3 data/scripts/extract_features_parallel.py 2016 --workers 4
  data/.venv/bin/python3 data/scripts/extract_features_parallel.py 2016 --dates 20160601,20160602   (test batch, no merge)
"""
import os
import csv
import argparse
import datetime
import multiprocessing as mp

import pandas as pd
import psutil

from grid_utils import load_boundaries
from imd_utils import imd_subdivision_daily
from era5_utils import (era5_subdivision_daily, build_era5_grid_map, era5_features_for_issue_date,
                         hourly_tp_subdivision_daily, merge_tp_into_daily)
from gefs_utils import gefs_daily_subdivision, build_gefs_grid_map
from download_gefs import download_date
from download_imd import download_year as download_imd_year
from atomic_io import atomic_to_parquet

OUT_DIR = "data/processed"
PARTS_DIR_TMPL = os.path.join(OUT_DIR, "_parts", "{year}")
GEFS_DIR = "data/raw/gefs"

MEMORY_PER_WORKER_BYTES = 500 * 1024 * 1024  # conservative: GEFS decode peak + pandas overhead
MAX_WORKERS = 10


def year_dates(year: int):
    d = datetime.date(year, 1, 1)
    end = datetime.date(year, 12, 31)
    while d <= end:
        yield d
        d += datetime.timedelta(days=1)


def choose_worker_count(override=None):
    if override:
        return override
    available = psutil.virtual_memory().available
    return max(1, min(MAX_WORKERS, available // MEMORY_PER_WORKER_BYTES))


# ---- worker-side globals, set once per worker process by _worker_init ----
_W = {}


def _worker_init(codes, subdivisions_by_code, imd_daily, era5_daily, gefs_grid_map, year):
    _W["codes"] = codes
    _W["subdivisions_by_code"] = subdivisions_by_code
    _W["imd_daily"] = imd_daily
    _W["era5_daily"] = era5_daily
    _W["gefs_grid_map"] = gefs_grid_map
    code_to_ij = {}
    for (i, j), code in gefs_grid_map.items():
        code_to_ij.setdefault(code, []).append((i, j))
    _W["gefs_code_to_ij"] = code_to_ij
    _W["year"] = year


def _process_one_date(init_date: datetime.date):
    year = _W["year"]
    codes = _W["codes"]
    subdivisions_by_code = _W["subdivisions_by_code"]
    imd_daily = _W["imd_daily"]
    era5_daily = _W["era5_daily"]
    gefs_grid_map = _W["gefs_grid_map"]
    gefs_code_to_ij = _W["gefs_code_to_ij"]

    date_str = init_date.strftime("%Y%m%d")
    part_path = os.path.join(PARTS_DIR_TMPL.format(year=year), f"{date_str}.parquet")
    if os.path.exists(part_path):
        return (date_str, "already_done", 0)

    # A single date's persistent failure (e.g. a network error that outlasts
    # download_date's own retries) must not kill the whole pool and lose
    # every other in-flight worker's progress -- report it and move on.
    # Already-written part files for other dates are unaffected either way.
    try:
        ok = download_date(init_date)
        if not ok:
            return (date_str, "no_gefs_file", 0)

        gefs_result = gefs_daily_subdivision(date_str, gefs_grid_map, gefs_code_to_ij)

        grib_path = os.path.join(GEFS_DIR, f"apcp_sfc_{date_str}00_c00.grib2")
        if os.path.exists(grib_path):
            os.remove(grib_path)

        if gefs_result is None:
            return (date_str, "decode_failed", 0)

        rows = []
        for code in codes:
            name = subdivisions_by_code[code]
            # -1 day shift: a forecast issued at 00Z on init_date can only
            # have seen ERA5 data through init_date - 1 (see era5_utils.py).
            era5_feats = era5_features_for_issue_date(era5_daily, init_date, code)
            for lead_day in range(1, 11):
                observed_date = init_date + datetime.timedelta(days=lead_day)
                forecast = gefs_result.get((lead_day, code))
                observed = imd_daily.get((observed_date, code))
                if forecast is None or observed is None:
                    continue
                if forecast != forecast or observed != observed:  # NaN check
                    continue
                rows.append({
                    "date": init_date.isoformat(),
                    "subdivision_code": code,
                    "subdivision_name": name,
                    "lead_day": lead_day,
                    "forecast_rain_mm": forecast,
                    "observed_rain_mm": observed,
                    "error_mm": forecast - observed,
                    "era5_msl_pa": era5_feats.get("msl"),
                    "era5_sp_pa": era5_feats.get("sp"),
                    "era5_d2m_k": era5_feats.get("d2m"),
                    "era5_t2m_k": era5_feats.get("t2m"),
                    "year": year,
                })

        df = pd.DataFrame(rows)
        atomic_to_parquet(df, part_path)
        return (date_str, "ok", len(rows))
    except Exception as e:
        print(f"[{year}] {date_str}: FAILED ({type(e).__name__}: {e}) -- skipped, will not be in the merged output", flush=True)
        return (date_str, f"error:{type(e).__name__}", 0)


def merge_parts(year: int) -> pd.DataFrame:
    parts_dir = PARTS_DIR_TMPL.format(year=year)
    files = sorted(f for f in os.listdir(parts_dir) if f.endswith(".parquet"))
    frames = [pd.read_parquet(os.path.join(parts_dir, f)) for f in files]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out_path = os.path.join(OUT_DIR, f"_features_{year}.parquet")
    atomic_to_parquet(combined, out_path)
    return combined


def main(year: int, workers_override=None, only_dates=None):
    geoms, codes = load_boundaries()
    subdivisions_by_code = {}
    with open("data/scripts/subdivisions.csv") as fh:
        for row in csv.DictReader(fh):
            subdivisions_by_code[row["subdivision_code"]] = row["subdivision_name"]

    print(f"[{year}] loading IMD (this year + next, for December lead-day spillover)...", flush=True)
    imd_daily, _ = imd_subdivision_daily(year, geoms=geoms, codes=codes)
    try:
        download_imd_year(year + 1)
        imd_next, _ = imd_subdivision_daily(year + 1, geoms=geoms, codes=codes)
        imd_daily.update(imd_next)
    except Exception as e:
        print(f"[{year}] could not load {year + 1} IMD for spillover ({e})", flush=True)

    print(f"[{year}] loading ERA5 features (this year + Dec of the previous year, "
          f"for the Jan 1 -1-day lookback)...", flush=True)
    era5_grid_map = build_era5_grid_map()
    era5_daily = era5_subdivision_daily(year, era5_grid_map)
    merge_tp_into_daily(era5_daily, hourly_tp_subdivision_daily(year, era5_grid_map))
    try:
        era5_prev = era5_subdivision_daily(year - 1, era5_grid_map)
        merge_tp_into_daily(era5_prev, hourly_tp_subdivision_daily(year - 1, era5_grid_map))
        era5_daily.update(era5_prev)  # era5_daily's own year's dates take precedence on overlap (none expected)
    except Exception as e:
        print(f"[{year}] no {year - 1} ERA5 for the Jan 1 lookback ({e}); "
              f"Jan 1 rows will have missing ERA5 features", flush=True)

    print(f"[{year}] preparing GEFS grid map...", flush=True)
    first_date = datetime.date(year, 1, 1)
    download_date(first_date)
    gefs_grid_map = build_gefs_grid_map()
    grib_path0 = os.path.join(GEFS_DIR, f"apcp_sfc_{first_date.strftime('%Y%m%d')}00_c00.grib2")
    if os.path.exists(grib_path0):
        os.remove(grib_path0)

    if only_dates:
        dates = [datetime.datetime.strptime(d, "%Y%m%d").date() for d in only_dates]
    else:
        dates = list(year_dates(year))

    parts_dir = PARTS_DIR_TMPL.format(year=year)
    os.makedirs(parts_dir, exist_ok=True)
    already = {f[:-8] for f in os.listdir(parts_dir) if f.endswith(".parquet")}
    todo = [d for d in dates if d.strftime("%Y%m%d") not in already]
    print(f"[{year}] {len(already)} dates already have part files, {len(todo)} to process", flush=True)

    n_workers = choose_worker_count(workers_override)
    avail_gb = psutil.virtual_memory().available / 1e9
    print(f"[{year}] available RAM: {avail_gb:.1f} GB -> using {n_workers} worker process(es)", flush=True)

    if not todo:
        print(f"[{year}] nothing to do", flush=True)
    else:
        with mp.Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=(codes, subdivisions_by_code, imd_daily, era5_daily, gefs_grid_map, year),
        ) as pool:
            n_done = 0
            statuses = {}
            failed_dates = []
            for date_str, status, n_rows in pool.imap_unordered(_process_one_date, todo):
                n_done += 1
                statuses[status] = statuses.get(status, 0) + 1
                if status.startswith("error"):
                    failed_dates.append(date_str)
                if n_done % 20 == 0 or n_done == len(todo):
                    print(f"[{year}] {n_done}/{len(todo)} done  (status counts so far: {statuses})", flush=True)
            if failed_dates:
                print(f"[{year}] {len(failed_dates)} date(s) FAILED and are missing from the output: "
                      f"{','.join(sorted(failed_dates))}", flush=True)
                print(f"[{year}] retry just these with: extract_features_parallel.py {year} "
                      f"--dates {','.join(sorted(failed_dates))}", flush=True)

    if not only_dates:
        combined = merge_parts(year)
        print(f"[{year}] merged: {len(combined)} total rows -> data/processed/_features_{year}.parquet", flush=True)
    else:
        print(f"[{year}] test batch done, part files left in {parts_dir} (not merged into the main file)", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("year", type=int)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--dates", type=str, default=None, help="comma-separated YYYYMMDD list, for testing")
    args = p.parse_args()
    only_dates = args.dates.split(",") if args.dates else None
    main(args.year, workers_override=args.workers, only_dates=only_dates)
