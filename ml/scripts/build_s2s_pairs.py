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
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML))
from vishwas_ml.ncmrwf_s2s import read_request, to_subdivisions  # noqa: E402
from vishwas_ml.subdivision_weights import build_weights, load_subdivisions  # noqa: E402

N216_WEIGHTS = ML / "data" / "weights_ncmrwf_s2s_n216.csv"


def imd_subdivision_daily(imd_dir, years, subs):
    """Area-weighted IMD daily rainfall per subdivision (date = IMD date)."""
    import imdlib as imd

    out = []
    w = None
    for y in years:
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
    return pd.concat(out, ignore_index=True).rename(columns={"time": "valid_date"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2s", nargs="+", required=True)
    ap.add_argument("--imd-dir", required=True)
    ap.add_argument("--shapefile", required=True, help="path without extension, e.g. .../indian_met_zones.v2")
    ap.add_argument("--era5", nargs="*", default=[], help="data-team parquet(s) with date, subdivision_code, era5_*")
    ap.add_argument("--out", default=str(ML / "data" / "ncmrwf_pairs.parquet"))
    a = ap.parse_args()

    subs = load_subdivisions(a.shapefile)
    grid = pd.concat([read_request(s) for s in a.s2s], ignore_index=True)
    grid = grid.drop_duplicates(["init_date", "lead_day", "latitude", "longitude"])
    w = pd.read_csv(N216_WEIGHTS) if N216_WEIGHTS.exists() else \
        build_weights(grid["latitude"].unique(), grid["longitude"].unique(), subs)
    fc = to_subdivisions(grid, w)
    fc["valid_date"] = fc["date"] + pd.to_timedelta(fc["lead_day"], "D")
    years = sorted(set(fc["valid_date"].dt.year))
    obs = imd_subdivision_daily(a.imd_dir, years, subs)
    pairs = fc.merge(obs, on=["valid_date", "subdivision_code"], how="left")
    missing = int(pairs["observed_rain_mm"].isna().sum())
    pairs["error_mm"] = pairs["forecast_rain_mm"] - pairs["observed_rain_mm"]
    if a.era5:
        era = pd.concat([pd.read_parquet(p) for p in a.era5], ignore_index=True)
        era["date"] = pd.to_datetime(era["date"])
        cols = [c for c in era.columns if c.startswith("era5_")]
        # ERA5 must be the issue-date state: one value per (date, subdivision), same for every lead.
        era = era.groupby(["date", "subdivision_code"], as_index=False)[cols].first()
        pairs = pairs.merge(era, on=["date", "subdivision_code"], how="left")
    pairs = pairs.drop(columns="valid_date").sort_values(["date", "subdivision_code", "lead_day"])
    pairs["year"] = pairs["date"].dt.year
    pairs.to_parquet(a.out, index=False)
    print(f"wrote {a.out}: {len(pairs):,} rows, {pairs['date'].nunique()} init dates "
          f"({pairs['date'].min().date()} .. {pairs['date'].max().date()}), {pairs['subdivision_code'].nunique()} "
          f"subdivisions; rows without IMD obs: {missing}")


if __name__ == "__main__":
    main()
