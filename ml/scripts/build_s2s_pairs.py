"""Build the pairs table from NCMRWF S2S downloads + IMD gridded rainfall (+ optional ERA5).

    python ml/scripts/build_s2s_pairs.py --s2s <zip-or-folder> [<zip-or-folder> ...] \
        --imd-dir <imdlib dir with rain/<year>.grd> --shapefile <.../indian_met_zones.v2> \
        [--era5 <data team parquet(s) with era5_* columns>] --out ml/data/ncmrwf_pairs.parquet

Output columns follow the data team's format (forecast_rain_mm, observed_rain_mm, error_mm,
era5_*), which vishwas_ml.schema reads directly. Point config.json data_path at the output.

Pairing (verified on 2015 against IMD: lead-1 correlation 0.77 at the correct day vs 0.62/0.54
on the neighbouring days): S2S file dayNN -> lead_day NN+1 -> IMD date init + lead_day.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML))
from vishwas_ml.ncmrwf_s2s import build_pairs, read_request, to_subdivisions  # noqa: E402
from vishwas_ml.subdivision_weights import build_weights, load_subdivisions  # noqa: E402

N216_WEIGHTS = ML / "data" / "weights_ncmrwf_s2s_n216.csv"


def imd_subdivision_daily(imd_dir, years, subs):
    """Area-weighted IMD daily rainfall per subdivision (date = IMD date)."""
    import imdlib as imd

    out = []
    w = None
    for y in years:
        if not (Path(imd_dir) / "rain" / f"{y}.grd").exists():
            print(f"  IMD {y} not in {imd_dir}; download with imdlib.get_data('rain', {y}, {y}, "
                  f"fn_format='yearwise', file_dir='{imd_dir}')")
            continue
        r = imd.open_data("rain", y, y, "yearwise", str(imd_dir)).get_xarray()["rain"]
        r = r.where(r != -999.0)
        if w is None:
            w = build_weights(r.lat.values, r.lon.values, subs, dlat=0.25, dlon=0.25)
            valid = r.notnull().any("time")
            LA, LO = np.meshgrid(r.lat.values, r.lon.values, indexing="ij")
            ok = pd.DataFrame({"latitude": LA.ravel(), "longitude": LO.ravel(), "v": valid.values.ravel()})
            w = w.merge(ok, on=["latitude", "longitude"])
            w = w[w["v"]].drop(columns="v")
        df = r.to_dataframe().reset_index().rename(columns={"lat": "latitude", "lon": "longitude"}).dropna()
        m = df.merge(w, on=["latitude", "longitude"])
        m["wr"] = m["rain"] * m["weight"]
        g = m.groupby(["time", "subdivision_code"])
        out.append((g["wr"].sum() / g["weight"].sum()).rename("observed_rain_mm").reset_index())
    return pd.concat(out, ignore_index=True).rename(columns={"time": "date"})


def _one_request(src):
    import warnings
    warnings.simplefilter("ignore")
    return to_subdivisions(read_request(src), pd.read_csv(N216_WEIGHTS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2s", nargs="*", default=[], help="S2S request zips or folders")
    ap.add_argument("--fc", default=None, help="subdivision forecast parquet: written after extraction, or read "
                    "instead of --s2s if it exists")
    ap.add_argument("--imd-dir", default=None, help="omit to stop after writing --fc")
    ap.add_argument("--shapefile", required=True, help="path without extension, e.g. .../indian_met_zones.v2")
    ap.add_argument("--era5", nargs="*", default=[], help="data-team parquet(s) with date, subdivision_code, era5_*")
    ap.add_argument("--out", default=str(ML / "data" / "ncmrwf_pairs.parquet"))
    a = ap.parse_args()

    subs = load_subdivisions(a.shapefile)
    if not N216_WEIGHTS.exists():
        g0 = read_request(a.s2s[0])
        build_weights(g0["latitude"].unique(), g0["longitude"].unique(), subs).to_csv(N216_WEIGHTS, index=False)
    if a.fc and Path(a.fc).exists() and not a.s2s:
        fc = pd.read_parquet(a.fc)
    else:
        workers = max(1, min(len(a.s2s), (os.cpu_count() or 2) - 1))
        with ProcessPoolExecutor(workers) as ex:  # one request (year) per worker, reduced to subdivisions there
            parts = list(ex.map(_one_request, a.s2s))
        fc = pd.concat(parts, ignore_index=True)
        fc = fc.groupby(["date", "subdivision_code", "lead_day"], as_index=False).first()
        if a.fc:
            fc.to_parquet(a.fc, index=False)
            print(f"wrote {a.fc}: {len(fc):,} subdivision forecast rows")
    if a.imd_dir is None:
        return
    valid = fc["date"] + pd.to_timedelta(fc["lead_day"], "D")
    years = sorted(set(valid.dt.year) | set((fc["date"] - pd.Timedelta(days=1)).dt.year))
    obs = imd_subdivision_daily(a.imd_dir, years, subs)
    pairs = build_pairs(fc, obs)
    if a.era5:
        era = pd.concat([pd.read_parquet(p) for p in a.era5], ignore_index=True)
        era["date"] = pd.to_datetime(era["date"])
        cols = [c for c in era.columns if c.startswith("era5_")]
        # ERA5 must be the issue-date state: one value per (date, subdivision), same for every lead.
        era = era.groupby(["date", "subdivision_code"], as_index=False)[cols].first()
        pairs = pairs.merge(era, on=["date", "subdivision_code"], how="left")
    pairs.to_parquet(a.out, index=False)
    missing = int(pairs["observed_rain_mm"].isna().sum())
    state = [c for c in ("mslp", "surface_pressure", "wind_u10", "wind_v10") if c in pairs]
    print(f"wrote {a.out}: {len(pairs):,} rows, {pairs['date'].nunique()} init dates "
          f"({pairs['date'].min().date()} .. {pairs['date'].max().date()}), {pairs['subdivision_code'].nunique()} "
          f"subdivisions, years {sorted(pairs['year'].unique().tolist())}")
    print(f"  rows without IMD obs: {missing}; weather-state columns: {state or 'none (rain-only download)'}; "
          f"previous-day IMD rain missing: {int(pairs['total_precipitation'].isna().sum())}")


if __name__ == "__main__":
    main()
