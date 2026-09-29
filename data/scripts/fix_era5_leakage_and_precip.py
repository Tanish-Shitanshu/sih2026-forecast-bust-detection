"""
Fixes two bugs Mohit found in the already-built bust_dataset.parquet:

Bug 1 (leakage): ERA5 features were attached at the forecast's own issue
date, but a 00Z forecast can only have seen data through issue_date - 1;
three of the issue date's own four synoptic snapshots (06/12/18Z) are in
that forecast's future. Fixed by rebuilding every ERA5 column via
era5_features_for_issue_date, which enforces the -1 day shift.

Bug 2 (precip scale): total_precipitation was summed from only 4 of a
day's 24 hourly ERA5 values (CDS delivers ERA5 tp as clean 1-hour
increments), undercounting the real daily total by ~5.5x. Fixed by
summing all 24 hourly values instead (hourly_tp_subdivision_daily), from
a separate download (download_era5_hourly_tp.py).

GEFS/IMD pairing (forecast_rain, observed_rain, error, is_bust,
trigger_reason) is untouched -- only the 7 ERA5 columns are rebuilt and
replaced.

Usage: data/.venv/bin/python3 data/scripts/fix_era5_leakage_and_precip.py 2015 2016 2017 2018 2019
"""
import sys
import datetime
import pandas as pd

from era5_utils import (era5_subdivision_daily, build_era5_grid_map,
                         era5_features_for_issue_date, hourly_tp_subdivision_daily,
                         merge_tp_into_daily)
from atomic_io import atomic_to_parquet

# internal era5_utils key -> the dataset's (already-target-schema) column name
RENAMED = {
    "msl": "mslp", "sp": "surface_pressure", "d2m": "dewpoint_2m", "t2m": "temp_2m",
    "u10": "wind_u10", "v10": "wind_v10", "tp": "total_precipitation",
}


def load_year(year, grid_map):
    yearly = era5_subdivision_daily(year, grid_map)
    merge_tp_into_daily(yearly, hourly_tp_subdivision_daily(year, grid_map))
    return yearly


def main(years):
    grid_map = build_era5_grid_map()

    era5_daily = {}
    for year in years:
        era5_daily.update(load_year(year, grid_map))

    # the earliest requested year's Jan 1 needs the previous year's Dec 31
    prev_year = min(years) - 1
    try:
        era5_daily.update({**load_year(prev_year, grid_map), **era5_daily})
    except Exception as e:
        print(f"no {prev_year} ERA5 for the earliest year's Jan-1 lookback ({e}); those rows' ERA5 columns will be NaN")

    df = pd.read_parquet("data/processed/bust_dataset.parquet")
    print(f"loaded bust_dataset.parquet: {len(df)} rows")

    unique_pairs = df[["date", "subdivision_code"]].drop_duplicates()
    print(f"{len(unique_pairs)} unique (date, subdivision_code) pairs to recompute")

    records = []
    for date_str, code in unique_pairs.itertuples(index=False):
        issue_date = datetime.date.fromisoformat(date_str)
        feats = era5_features_for_issue_date(era5_daily, issue_date, code)
        rec = {"date": date_str, "subdivision_code": code}
        for internal_key, col in RENAMED.items():
            rec[col] = feats.get(internal_key)
        records.append(rec)
    replacement = pd.DataFrame(records)

    missing = replacement["mslp"].isna().sum()
    print(f"pairs with no ERA5 match (expected: only issue dates needing data before our earliest downloaded year): {missing}")
    if missing:
        print(replacement[replacement["mslp"].isna()]["date"].value_counts())

    old_cols = list(RENAMED.values())
    before = len(df)
    df = df.drop(columns=old_cols).merge(replacement, on=["date", "subdivision_code"], how="left")
    assert len(df) == before, f"merge changed row count: {before} -> {len(df)}"

    atomic_to_parquet(df, "data/processed/bust_dataset.parquet")
    print(f"\nwrote data/processed/bust_dataset.parquet: {len(df)} rows")
    print("columns:", list(df.columns))


if __name__ == "__main__":
    years = [int(y) for y in sys.argv[1:]]
    if not years:
        raise SystemExit("usage: fix_era5_leakage_and_precip.py YEAR [YEAR ...]")
    main(years)
