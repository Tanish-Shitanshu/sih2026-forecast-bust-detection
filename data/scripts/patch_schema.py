"""
One-time patch: adds the previously-unextracted ERA5 wind (u10, v10) and
total precipitation (tp) features to the already-built bust_dataset, and
renames every column to the target handoff schema. Doesn't touch GEFS/IMD
at all -- those pairings are already correct in bust_dataset.parquet, this
only needs a (date, subdivision_code) join for the 3 new ERA5 columns.

Usage: data/.venv/bin/python3 data/scripts/patch_schema.py 2015 2016 2017 2018 2019
"""
import sys
import pandas as pd

from era5_utils import era5_subdivision_daily, build_era5_grid_map
from atomic_io import atomic_to_parquet

RENAME_MAP = {
    "forecast_rain_mm": "forecast_rain",
    "observed_rain_mm": "observed_rain",
    "error_mm": "error",
    "era5_msl_pa": "mslp",
    "era5_sp_pa": "surface_pressure",
    "era5_d2m_k": "dewpoint_2m",
    "era5_t2m_k": "temp_2m",
}


def main(years):
    df = pd.read_parquet("data/processed/bust_dataset.parquet")
    print(f"loaded bust_dataset.parquet: {len(df)} rows, {len(df.columns)} columns")

    grid_map = build_era5_grid_map()
    new_rows = []
    for year in years:
        daily = era5_subdivision_daily(year, grid_map)
        for (date, code), feats in daily.items():
            new_rows.append({
                "date": date.isoformat(),
                "subdivision_code": code,
                "wind_u10": feats.get("u10"),
                "wind_v10": feats.get("v10"),
                "total_precipitation": feats.get("tp"),
            })
        print(f"[{year}] extracted u10/v10/tp for {len(daily)} (date, subdivision) pairs")

    era5_extra = pd.DataFrame(new_rows).drop_duplicates(subset=["date", "subdivision_code"])
    print(f"era5_extra: {len(era5_extra)} unique (date, subdivision_code) rows")

    before = len(df)
    df = df.merge(era5_extra, on=["date", "subdivision_code"], how="left")
    assert len(df) == before, f"merge changed row count: {before} -> {len(df)}"

    missing = df["wind_u10"].isna().sum()
    if missing:
        print(f"WARNING: {missing} rows have no matching (date, subdivision_code) in the ERA5 extras")
    else:
        print("all rows matched -- no missing wind_u10/wind_v10/total_precipitation")

    df = df.rename(columns=RENAME_MAP)

    target_schema = [
        "date", "subdivision_code", "lead_day", "forecast_rain", "observed_rain",
        "error", "is_bust", "trigger_reason", "mslp", "surface_pressure",
        "dewpoint_2m", "temp_2m", "wind_u10", "wind_v10", "total_precipitation",
    ]
    extra_cols = ["subdivision_name", "magnitude_threshold_mm", "year", "split"]
    df = df[target_schema + extra_cols]

    missing_from_target = [c for c in target_schema if c not in df.columns]
    assert not missing_from_target, f"still missing: {missing_from_target}"

    atomic_to_parquet(df, "data/processed/bust_dataset.parquet")
    print(f"\nwrote data/processed/bust_dataset.parquet: {len(df)} rows, {len(df.columns)} columns")
    print("target schema columns present:", target_schema)
    print("additional columns kept:", extra_cols)


if __name__ == "__main__":
    years = [int(y) for y in sys.argv[1:]]
    if not years:
        raise SystemExit("usage: patch_schema.py YEAR [YEAR ...]")
    main(years)
