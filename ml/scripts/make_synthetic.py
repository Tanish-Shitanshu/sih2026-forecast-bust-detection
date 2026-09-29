"""Generate a synthetic pairs table with the exact schema of the real dataset.

    python ml/scripts/make_synthetic.py [--out ml/data/synthetic_pairs.parquet] [--seed 0]

How it is built (all made up, but physically shaped so downstream components have
something real to find):
  * A daily "true" weather series per subdivision (2020-2023): seasonal rain climatology
    (SW monsoon, NE monsoon, western disturbances, pre-monsoon storms) modulated by a
    persistent disturbance index z (regional + local AR(1) plus depression/cyclone shocks).
  * ERA5-like issue-date state from z and the season: lower pressure and higher dewpoint
    when disturbed, a heat-wave index for temperature, monsoon wind reversal.
    ERA5 values are the issue-date state, so they are identical across the 10 lead days.
  * Forecasts for valid date = issue date + lead: the observation seen through a lead-
    dependent error (log-normal amplitude error, timing displacement, drift to climatology,
    a drizzle bias). Errors grow when the issue-date state is disturbed and moist, so the
    ERA5 columns carry real signal about bust risk.
  * observed_rain is the valid-date value, so it is consistent across issue dates.
  * Labels use the locked definition with thresholds fit on the training years only.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vishwas_ml.synthetic import generate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "synthetic_pairs.parquet"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=3, help="days between issue dates")
    a = ap.parse_args()
    df = generate(seed=a.seed, every=a.every)
    df.to_parquet(a.out, index=False)
    by_lead = df.groupby("lead_day")["is_bust"].mean().round(3).to_dict()
    print(f"wrote {a.out}: {len(df):,} rows, {df['date'].nunique()} issue dates, "
          f"{df['subdivision_code'].nunique()} subdivisions")
    print(f"bust rate {df['is_bust'].mean():.3f}; by lead {by_lead}")
    print("trigger reasons:", df["trigger_reason"].value_counts().to_dict())


if __name__ == "__main__":
    main()
