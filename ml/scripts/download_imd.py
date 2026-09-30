"""Download IMD gridded daily rainfall (0.25 deg, one .grd per year) with imdlib, with retries.

    python ml/scripts/download_imd.py --years 1992-2025 --out <dir>      # files land in <dir>/rain/YYYY.grd

Only needed to rebuild observations from scratch: the per-subdivision result for 1992-2025 is
already committed as ml/data/external/imd_subdivision_daily_1992_2025.parquet. The IMD server is
flaky, so each year is retried; a complete year file is about 25 MB.
"""
import argparse
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="1992-2025", help="e.g. 2017-2025 or 2019")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tries", type=int, default=6)
    a = ap.parse_args()
    import imdlib as imd

    lo, _, hi = a.years.partition("-")
    for y in range(int(lo), int(hi or lo) + 1):
        f = Path(a.out) / "rain" / f"{y}.grd"
        for _ in range(a.tries):
            if f.exists() and f.stat().st_size > 20_000_000:
                break
            try:
                imd.get_data("rain", y, y, fn_format="yearwise", file_dir=a.out)
            except Exception as e:  # server drops connections; retry
                print(f"  {y}: {e}")
            time.sleep(3)
        print(y, f.stat().st_size if f.exists() else "missing")


if __name__ == "__main__":
    main()
