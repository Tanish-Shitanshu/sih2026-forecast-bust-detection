"""Reader for NCMRWF S2S reforecast precipitation downloaded from rds.ncmrwf.gov.in.

What a download looks like (checked on the 2015-07-01 test request, 2026-09-29):
  * a zip of files APCP-sfc_IC<YYYYMMDD>_day<NN>.nc, one per Forecast Day (day00 = "T")
  * variable `apcp` on (t, latitude, longitude); t = init date + NN days
  * N216 grid: 0.8333 deg longitude x 0.5556 deg latitude
  * values are daily totals in mm, even though the attribute says "kg m-2 s-1"
    (July NE-India means 10-40 mm; x86400 would give ~1e6 mm/day). check_units() guards this.
  * one field per day: the portal offers no ensemble-member choice

Lead-day convention used here: file dayNN covers hours 24*NN..24*(NN+1) after the 00Z init,
i.e. the 24 h ending 00Z on init+NN+1. IMD's day D ends 03Z on D, so file dayNN pairs with IMD
date init+NN+1, which is our lead_day = NN+1 (valid date = issue date + lead_day).
So lead days 1-10 need files day00..day09; day10 is not used.

Portal gotcha: the coordinate boxes are ordered NORTH, SOUTH, EAST, WEST. Swapping east and west
returns the rest of the globe instead of India; read_request() raises if 68-98E is missing.
"""
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

FILE_RE = re.compile(r"APCP-sfc_IC(\d{8})_day(\d{2})\.nc$")
INDIA = {"lat": (6.0, 38.0), "lon": (68.0, 98.0)}


def _files(src):
    """Paths of the day files inside a zip (extracted next to it) or a folder."""
    src = Path(src)
    if src.suffix == ".zip":
        out = src.with_suffix("")
        with zipfile.ZipFile(src) as z:
            z.extractall(out)
        src = out
    return sorted(p for p in src.rglob("*.nc") if FILE_RE.search(p.name))


def check_units(values):
    """Daily totals in mm should have a tropical mean of a few mm and a max of at most a few hundred."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size and (v.max() > 2000 or v.max() < 1e-3):
        raise ValueError(f"S2S precipitation max {v.max():.3g} is not a plausible daily total in mm; "
                         "check the units before using this file")


def read_request(src, max_day=9):
    """Tidy frame: init_date, lead_day (1..max_day+1), latitude, longitude (-180..180), forecast_rain_mm."""
    import xarray as xr

    rows = []
    for f in _files(src):
        init, nn = FILE_RE.search(f.name).groups()
        nn = int(nn)
        if nn > max_day:
            continue
        with xr.open_dataset(f) as ds:
            a = ds["apcp"].isel(t=0).load()
        lon = ((a["longitude"].values + 180) % 360) - 180
        inside = (lon >= INDIA["lon"][0]) & (lon <= INDIA["lon"][1])
        if inside.sum() < 30:
            raise ValueError(f"{f.name} covers only {inside.sum()} longitudes inside 68-98E; the request's EAST and "
                             "WEST boxes were probably swapped (form order is NORTH, SOUTH, EAST, WEST)")
        lat = a["latitude"].values
        keep_lat = (lat >= INDIA["lat"][0]) & (lat <= INDIA["lat"][1])
        v = a.values[np.ix_(keep_lat, inside)]
        check_units(v)
        LA, LO = np.meshgrid(lat[keep_lat], lon[inside], indexing="ij")
        rows.append(pd.DataFrame({"init_date": pd.Timestamp(init), "lead_day": nn + 1,
                                  "latitude": LA.ravel(), "longitude": LO.ravel(),
                                  "forecast_rain_mm": np.maximum(v.ravel(), 0.0)}))
    if not rows:
        raise FileNotFoundError(f"no APCP-sfc_IC*_dayNN.nc files in {src}")
    return pd.concat(rows, ignore_index=True)


def to_subdivisions(grid, weights):
    """Area-weighted subdivision means.

    grid: output of read_request. weights: the data pipeline's cell -> subdivision table with
    columns latitude, longitude, subdivision_code, weight (fraction of the S2S cell inside the
    subdivision x cell area; built once for the N216 grid, like the existing GEFS/ERA5 masks).
    Returns date, subdivision_code, lead_day, forecast_rain_mm, the forecast columns of the
    pairs table (date = issue/init date)."""
    w = weights[["latitude", "longitude", "subdivision_code", "weight"]].copy()
    g = grid.copy()
    for d in (g, w):  # float32 grid coordinates vs a mask built elsewhere: join on rounded values
        d["latitude"] = d["latitude"].round(3)
        d["longitude"] = d["longitude"].round(3)
    m = g.merge(w, on=["latitude", "longitude"], how="inner")
    if m.empty:
        raise ValueError("no S2S grid cell matched the weights table; build the weights on the N216 grid")
    m["wr"] = m["forecast_rain_mm"] * m["weight"]
    out = m.groupby(["init_date", "subdivision_code", "lead_day"]).agg(wr=("wr", "sum"), w=("weight", "sum"))
    out["forecast_rain_mm"] = out["wr"] / out["w"]
    out = out.reset_index().rename(columns={"init_date": "date"})
    return out[["date", "subdivision_code", "lead_day", "forecast_rain_mm"]]


def request_plan(years=range(1993, 2016), init_days=("01", "09", "17", "25")):
    """Portal requests for the full archive. On the form, Year and Initialization Day are single
    choice but Month is multi-select, so one request = one year x one init day x all 12 months,
    Forecast Day T..T+9: 23 x 4 = 92 requests."""
    return pd.DataFrame([{"year": y, "init_day": d, "months": "all 12", "forecast_days": "T..T+9",
                          "variable": "Total Precipitation Amount", "north": 38, "south": 6, "east": 98, "west": 68}
                         for y in years for d in init_days])
