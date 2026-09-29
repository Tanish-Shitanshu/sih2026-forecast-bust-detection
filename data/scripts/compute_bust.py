"""
Combines all per-year _features_{year}.parquet files and applies the LOCKED
bust definition:

  bust = (|forecast_rain - observed_rain| >= max(25mm, that subdivision's
          95th-percentile rain error FOR THAT SPECIFIC LEAD DAY, fit on
          training years only))
         OR (rain/no-rain category call was wrong)

The magnitude threshold is fit per (subdivision_code, lead_day) using ONLY
rows whose year is in TRAIN_YEARS, then applied to every row (train and
test) — this is what "fit on training years only" means here. The
rain/no-rain category call uses IMD's standard "rainy day" cutoff of
2.5mm/24h on both the forecast and observed values.

Usage: data/.venv/bin/python3 data/scripts/compute_bust.py 2015 2016 2017 2018 2019
"""
import sys
import glob
import pandas as pd
import numpy as np

OUT_DIR = "data/processed"
TRAIN_YEARS = {2015, 2016, 2017, 2018}
TEST_YEARS = {2019}
RAIN_CATEGORY_THRESHOLD_MM = 2.5  # IMD's standard rainy-day cutoff
MIN_MAGNITUDE_MM = 25.0


def main(years):
    frames = []
    for y in years:
        path = f"{OUT_DIR}/_features_{y}.parquet"
        df = pd.read_parquet(path)
        frames.append(df)
        print(f"loaded {path}: {len(df)} rows")
    df = pd.concat(frames, ignore_index=True)

    df["split"] = np.where(df["year"].isin(TRAIN_YEARS), "train",
                    np.where(df["year"].isin(TEST_YEARS), "test", "unused"))
    df = df[df["split"] != "unused"].copy()

    # --- magnitude threshold: 95th percentile of |error|, per (subdivision, lead_day), train rows only
    train = df[df["split"] == "train"]
    group_sizes = train.groupby(["subdivision_code", "lead_day"]).size()
    small_groups = group_sizes[group_sizes < 30]
    if len(small_groups):
        print(f"WARNING: {len(small_groups)} (subdivision, lead_day) groups have < 30 training rows; "
              f"their 95th-percentile threshold may be noisy.")

    pctl95 = (train.assign(abs_error=train["error_mm"].abs())
                    .groupby(["subdivision_code", "lead_day"])["abs_error"]
                    .quantile(0.95)
                    .rename("pctl95_abs_error_mm"))
    threshold = pctl95.apply(lambda v: max(MIN_MAGNITUDE_MM, v)).rename("magnitude_threshold_mm")

    df = df.merge(threshold.reset_index(), on=["subdivision_code", "lead_day"], how="left")
    if df["magnitude_threshold_mm"].isna().any():
        missing = df[df["magnitude_threshold_mm"].isna()][["subdivision_code", "lead_day"]].drop_duplicates()
        raise SystemExit(f"No training-derived threshold for these (subdivision, lead_day) pairs:\n{missing}")

    # --- locked bust formula
    magnitude_bust = df["error_mm"].abs() >= df["magnitude_threshold_mm"]
    forecast_category = df["forecast_rain_mm"] >= RAIN_CATEGORY_THRESHOLD_MM
    observed_category = df["observed_rain_mm"] >= RAIN_CATEGORY_THRESHOLD_MM
    category_wrong = forecast_category != observed_category

    df["is_bust"] = magnitude_bust | category_wrong
    df["trigger_reason"] = np.select(
        [magnitude_bust & category_wrong, magnitude_bust & ~category_wrong, ~magnitude_bust & category_wrong],
        ["both", "magnitude", "category"],
        default="none",
    )

    df = df[[
        "date", "subdivision_code", "subdivision_name", "lead_day",
        "forecast_rain_mm", "observed_rain_mm", "error_mm",
        "is_bust", "trigger_reason", "magnitude_threshold_mm",
        "era5_msl_pa", "era5_sp_pa", "era5_d2m_k", "era5_t2m_k",
        "year", "split",
    ]]

    out_parquet = f"{OUT_DIR}/bust_dataset.parquet"
    out_csv = f"{OUT_DIR}/bust_dataset.csv"
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)

    print(f"\nWrote {out_parquet} and {out_csv}: {len(df)} rows")
    print(f"is_bust rate: {df['is_bust'].mean():.1%}")
    print("trigger_reason breakdown:")
    print(df["trigger_reason"].value_counts())
    print("\nsplit breakdown:")
    print(df["split"].value_counts())


if __name__ == "__main__":
    years = [int(y) for y in sys.argv[1:]]
    if not years:
        raise SystemExit("usage: compute_bust.py YEAR [YEAR ...]")
    main(years)
