"""
Builds the raw (unlabeled) forecast-vs-observed pairing table for one full
calendar year (January 1 - December 31): every (init_date, subdivision,
lead_day 1-10) combination, with forecast_rain_mm (GEFS), observed_rain_mm
(IMD), error_mm, and a few basic ERA5 features at the init date. Bust
flags are NOT computed here (see compute_bust.py) since the magnitude
threshold must be fit across all years' training data at once.

Full-year, not monsoon-only: bust categories in scope (heat waves, western
disturbances, cyclones) aren't monsoon-exclusive, and this matches the
already-downloaded full 12-month ERA5 pull.

Downloads any missing GEFS files as it goes, and DELETES each GEFS GRIB2
file immediately after extracting its features — only the aggregated
subdivision-level numbers are needed downstream, and a full 5-year x
12-month GEFS pull (~45GB) doesn't fit comfortably otherwise. Re-run this
script to re-derive features; the raw file is just re-downloaded.

A late-December init date's lead days spill into next January, so this
also loads (and downloads if needed) next year's IMD file for that.

Usage: data/.venv/bin/python3 data/scripts/extract_features.py 2015
"""
import sys
import os
import csv
import datetime
import pandas as pd

from grid_utils import load_boundaries
from imd_utils import imd_subdivision_daily
from era5_utils import era5_subdivision_daily, build_era5_grid_map
from gefs_utils import gefs_daily_subdivision, build_gefs_grid_map
from download_gefs import download_date
from download_imd import download_year as download_imd_year
from atomic_io import atomic_to_parquet

OUT_DIR = "data/processed"
GEFS_DIR = "data/raw/gefs"


def year_dates(year: int):
    d = datetime.date(year, 1, 1)
    end = datetime.date(year, 12, 31)
    while d <= end:
        yield d
        d += datetime.timedelta(days=1)


def main(year: int):
    geoms, codes = load_boundaries()
    subdivisions_by_code = {}
    with open("data/scripts/subdivisions.csv") as fh:
        for row in csv.DictReader(fh):
            subdivisions_by_code[row["subdivision_code"]] = row["subdivision_name"]

    print(f"[{year}] loading IMD observed rainfall (this year + next, for December lead-day spillover)...")
    imd_daily, imd_grid_map = imd_subdivision_daily(year, geoms=geoms, codes=codes)
    try:
        download_imd_year(year + 1)
        imd_daily_next, _ = imd_subdivision_daily(year + 1, geoms=geoms, codes=codes)
        imd_daily.update(imd_daily_next)
    except Exception as e:
        print(f"[{year}] could not load {year + 1} IMD for December spillover ({e}); "
              f"late-December lead days beyond {year + 1}-01-xx will be dropped as missing")

    print(f"[{year}] loading ERA5 features...")
    era5_grid_map = build_era5_grid_map()
    era5_daily = era5_subdivision_daily(year, era5_grid_map)

    print(f"[{year}] preparing GEFS grid map...")
    # ensure at least one GEFS file exists locally to read the grid from
    first_date = datetime.date(year, 1, 1)
    download_date(first_date)
    gefs_grid_map = build_gefs_grid_map()
    gefs_code_to_ij = {}
    for (i, j), code in gefs_grid_map.items():
        gefs_code_to_ij.setdefault(code, []).append((i, j))

    out_path = os.path.join(OUT_DIR, f"_features_{year}.parquet")
    existing_df = None
    done_dates = set()
    if os.path.exists(out_path):
        existing_df = pd.read_parquet(out_path)
        done_dates = set(existing_df["date"].unique())
        print(f"[{year}] resuming: {len(done_dates)} init dates already in {out_path}")

    def checkpoint(rows):
        new_df = pd.DataFrame(rows)
        combined = pd.concat([existing_df, new_df], ignore_index=True) if existing_df is not None and len(existing_df) else new_df
        atomic_to_parquet(combined, out_path)
        return combined

    rows = []
    dates = list(year_dates(year))
    for n, init_date in enumerate(dates, 1):
        date_str = init_date.strftime("%Y%m%d")
        if init_date.isoformat() in done_dates:
            continue
        ok = download_date(init_date)
        if not ok:
            print(f"[{year}] {date_str}: no GEFS file, skipping this init date")
            continue
        gefs_result = gefs_daily_subdivision(date_str, gefs_grid_map, gefs_code_to_ij)

        grib_path = os.path.join(GEFS_DIR, f"apcp_sfc_{date_str}00_c00.grib2")
        idx_path = grib_path + ".idx" if os.path.exists(grib_path + ".idx") else None
        if os.path.exists(grib_path):
            os.remove(grib_path)
        if idx_path:
            os.remove(idx_path)

        if gefs_result is None:
            continue

        for code in codes:
            name = subdivisions_by_code[code]
            era5_feats = era5_daily.get((init_date, code), {})
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
        if n % 20 == 0 or n == len(dates):
            print(f"[{year}] processed {n}/{len(dates)} init dates this run, {len(rows)} new rows so far")
        if n % 20 == 0:
            existing_df = checkpoint(rows)
            rows = []
            print(f"[{year}] checkpointed: {len(existing_df)} total rows saved to {out_path}")

    final_df = checkpoint(rows)
    print(f"[{year}] wrote {out_path}: {len(final_df)} total rows")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: extract_features.py YEAR")
    main(int(sys.argv[1]))
