"""Build the pairs table for NCMRWF's operational NEPS forecasts (TIGGE) + IMD observations.

    python ml/scripts/build_neps_pairs.py --parts <dir>/parts --imd-dir <imdlib dir> \
        --shapefile <.../indian_met_zones.v2> --out ml/data/ncmrwf_neps_pairs.parquet

Input: the per-date subdivision files written by scripts/fetch_tigge_ncmrwf.py (lead_day 0 =
issue-time state, 1..10 = daily rain). Output: the pairs table in the data team's column format
(forecast_rain_mm, observed_rain_mm, error_mm, state columns, total_precipitation = IMD rain on
issue date - 1), which vishwas_ml.schema reads directly.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML))
sys.path.insert(0, str(ML / "scripts"))
from build_s2s_pairs import imd_subdivision_daily  # noqa: E402
from vishwas_ml.ncmrwf_tigge import build_pairs  # noqa: E402
from vishwas_ml.subdivision_weights import load_subdivisions  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True)
    ap.add_argument("--obs", default=str(ML / "data" / "external" / "imd_subdivision_daily_1992_2025.parquet"),
                    help="precomputed IMD subdivision daily rain (date, subdivision_code, observed_rain_mm)")
    ap.add_argument("--imd-dir", help="rebuild observations from imdlib .grd files instead of --obs")
    ap.add_argument("--shapefile", default=str(ML / "data" / "external" / "indian_met_zones" / "indian_met_zones.v2"))
    ap.add_argument("--out", default=str(ML / "data" / "ncmrwf_neps_pairs.parquet"))
    a = ap.parse_args()
    files = sorted(Path(a.parts).glob("*.parquet"))
    fc = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    fc["date"] = pd.to_datetime(fc["date"])
    years = sorted(set(fc["date"].dt.year) | set((fc["date"] + pd.Timedelta(days=10)).dt.year)
                   | set((fc["date"] - pd.Timedelta(days=1)).dt.year))
    if a.imd_dir:
        obs = imd_subdivision_daily(a.imd_dir, years, load_subdivisions(a.shapefile))
    else:
        obs = pd.read_parquet(a.obs)
        obs["date"] = pd.to_datetime(obs["date"])
    pairs = build_pairs(fc, obs)
    n_all = len(pairs)
    pairs = pairs[pairs["observed_rain_mm"].notna() & pairs["forecast_rain_mm"].notna()]
    pairs.to_parquet(a.out, index=False)
    print(f"wrote {a.out}: {len(pairs):,} rows ({n_all - len(pairs):,} dropped without IMD/forecast), "
          f"{pairs['date'].nunique()} issue dates {pairs['date'].min().date()} .. {pairs['date'].max().date()}, "
          f"years {sorted(pairs['year'].unique().tolist())}")


if __name__ == "__main__":
    main()
