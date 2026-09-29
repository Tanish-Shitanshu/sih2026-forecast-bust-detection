"""Reader for NCMRWF S2S reforecast downloads from rds.ncmrwf.gov.in.

What a download looks like (checked on real requests, 2026-09-29):
  * a zip of files <VAR>_IC<YYYYMMDD>_day<NN>.nc, one per variable and Forecast Day (day00 = "T"),
    e.g. APCP-sfc_IC20150701_day00.nc
  * one variable per file on (t, latitude, longitude); t = init date + NN days
  * N216 grid: 0.8333 deg longitude x 0.5556 deg latitude, float32 coordinates
  * precipitation `apcp` is a daily total in mm, although its attribute says "kg m-2 s-1"
    (July NE-India means 10-40 mm; x86400 would give ~1e6 mm/day). check_units() guards this.
  * one field per day: the portal offers no ensemble-member choice

Other single-level variables (MSLP, surface pressure, 10 m winds) are recognised from their
NetCDF name / standard_name, so their exact file names do not matter. Every S2S field is model
output available when the forecast is issued, so all of them are leak-free features.

Lead days (verified against IMD 2015: day00 correlates 0.77 with IMD date init+1, vs 0.62 on
init and 0.54 on init+2): file dayNN covers the 24 h after the 00Z init + NN days, so it pairs
with IMD date init+NN+1, which is our lead_day = NN+1. Lead days 1-10 need day00..day09.

Portal gotcha: the coordinate boxes are ordered NORTH, SOUTH, EAST, WEST. Swapping east and west
returns the rest of the globe instead of India; read_request() raises if 68-98E is missing.
"""
import re
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

FILE_RE = re.compile(r"^(?P<var>.+?)_IC(?P<init>\d{8})_day(?P<nn>\d{2})\.nc$")
INDIA = {"lat": (6.0, 38.0), "lon": (68.0, 98.0)}

# output column -> (lower-case names, standard_names) that identify it
VARIABLES = {
    "forecast_rain_mm": ({"apcp", "precip", "tp", "pr", "prate"}, {"precipitation_flux", "precipitation_amount"}),
    "mslp": ({"prmsl", "msl", "mslp", "psl", "pmsl"}, {"air_pressure_at_sea_level", "air_pressure_at_mean_sea_level"}),
    "surface_pressure": ({"pres", "sp", "ps", "psfc"}, {"surface_air_pressure"}),
    "wind_u10": ({"ugrd", "u10", "uas", "u"}, {"eastward_wind"}),
    "wind_v10": ({"vgrd", "v10", "vas", "v"}, {"northward_wind"}),
}
PLAUSIBLE = {  # allowed range of the field's largest absolute value over India
    "forecast_rain_mm": (1e-3, 2000.0),
    "mslp": (500.0, 1.2e5),          # hPa or Pa both accepted; features are z-scored
    "surface_pressure": (500.0, 1.2e5),
    "wind_u10": (0.1, 80.0),
    "wind_v10": (0.1, 80.0),
}


def _files(src):
    """Paths of the S2S day files inside a zip (extracted next to it) or a folder."""
    src = Path(src)
    if src.suffix == ".zip":
        out = src.with_suffix("")
        with zipfile.ZipFile(src) as z:
            z.extractall(out)
        src = out
    return sorted(p for p in src.rglob("*.nc") if FILE_RE.match(p.name))


def identify(da, file_var=""):
    """Which output column a NetCDF variable is, or None."""
    names = {str(da.name).lower(), str(da.attrs.get("name", "")).lower(), file_var.split("-")[0].lower()}
    std = str(da.attrs.get("standard_name", "")).lower()
    for col, (aliases, stds) in VARIABLES.items():
        if names & aliases or std in stds:
            return col
    return None


def check_units(values, col="forecast_rain_mm"):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    lo, hi = PLAUSIBLE[col]
    if v.size and not (lo <= np.abs(v).max() <= hi):
        raise ValueError(f"S2S {col} max {np.abs(v).max():.3g} is outside the plausible range {lo:g}..{hi:g}; "
                         "check the units before using this file")


def read_request(src, max_day=9):
    """Tidy wide frame: init_date, lead_day (1..max_day+1), latitude, longitude (-180..180) and one
    column per recognised variable (forecast_rain_mm, mslp, surface_pressure, wind_u10, wind_v10)."""
    import xarray as xr

    by_col = {}
    for f in _files(src):
        m = FILE_RE.match(f.name)
        nn = int(m["nn"])
        if nn > max_day:
            continue
        with xr.open_dataset(f) as ds:
            for name in ds.data_vars:
                da = ds[name]
                if "latitude" not in da.dims or "longitude" not in da.dims:
                    continue
                col = identify(da, m["var"])
                if col is None:
                    warnings.warn(f"{f.name}: unrecognised variable {name!r} skipped")
                    continue
                a = da.isel({d: 0 for d in da.dims if d not in ("latitude", "longitude")}).load()
                lon = ((a["longitude"].values.astype("float64") + 180) % 360) - 180
                inside = (lon >= INDIA["lon"][0]) & (lon <= INDIA["lon"][1])
                if inside.sum() < 30:
                    raise ValueError(f"{f.name} covers only {inside.sum()} longitudes inside 68-98E; the request's EAST "
                                     "and WEST boxes were probably swapped (form order is NORTH, SOUTH, EAST, WEST)")
                lat = a["latitude"].values.astype("float64")
                keep = (lat >= INDIA["lat"][0]) & (lat <= INDIA["lat"][1])
                v = a.values[np.ix_(keep, inside)].astype("float64")
                check_units(v, col)
                if col == "forecast_rain_mm":
                    v = np.maximum(v, 0.0)
                LA, LO = np.meshgrid(lat[keep], lon[inside], indexing="ij")
                by_col.setdefault(col, []).append(pd.DataFrame({
                    "init_date": pd.Timestamp(m["init"]), "lead_day": nn + 1,
                    "latitude": LA.ravel(), "longitude": LO.ravel(), col: v.ravel()}))
    if not by_col:
        raise FileNotFoundError(f"no <VAR>_IC<date>_dayNN.nc files in {src}")
    keys = ["init_date", "lead_day", "latitude", "longitude"]
    out = None
    for col, dfs in by_col.items():
        d = pd.concat(dfs, ignore_index=True).drop_duplicates(keys)
        out = d if out is None else out.merge(d, on=keys, how="outer")
    return out.sort_values(keys, kind="stable").reset_index(drop=True)


def to_subdivisions(grid, weights):
    """Area-weighted subdivision means of every value column in `grid`.

    grid: output of read_request. weights: cell -> subdivision table with columns latitude,
    longitude, subdivision_code, weight (see subdivision_weights.build_weights).
    Returns date (= init/issue date), subdivision_code, lead_day and one column per variable."""
    keys = ["init_date", "lead_day", "latitude", "longitude"]
    values = [c for c in grid.columns if c not in keys]
    w = weights[["latitude", "longitude", "subdivision_code", "weight"]].copy()
    g = grid.copy()
    for d in (g, w):  # float32 NetCDF coordinates vs a float64 table: cast, then join on rounded values
        d["latitude"] = d["latitude"].astype("float64").round(3)
        d["longitude"] = d["longitude"].astype("float64").round(3)
    grp = ["init_date", "subdivision_code", "lead_day"]
    out = None
    for c in values:
        # Each variable is averaged over its own grid: UM winds sit on a grid staggered half a
        # cell in longitude from rain and pressure, so the weights table carries both grids.
        gc = g.loc[g[c].notna(), grp[:1] + ["lead_day", "latitude", "longitude", c]]
        m = gc.merge(w, on=["latitude", "longitude"], how="inner")
        if m.empty:
            raise ValueError(f"no {c} grid cell matched the weights table; build weights for this variable's grid")
        lost = set(w["subdivision_code"]) - set(m["subdivision_code"])
        if lost:
            raise ValueError(f"{c}: weights for {sorted(lost)} matched no grid cell; grid and weights coordinates differ")
        m["wv"] = m[c] * m["weight"]
        r = m.groupby(grp).agg(wv=("wv", "sum"), w=("weight", "sum"))
        r = (r["wv"] / r["w"]).rename(c).reset_index()
        out = r if out is None else out.merge(r, on=grp, how="outer")
    out = out.rename(columns={"init_date": "date"})
    return out[["date", "subdivision_code", "lead_day"] + values]


def request_plan(years=range(1993, 2016), init_days=("01", "09", "17", "25")):
    """Portal requests for the full archive. On the form, Year and Initialization Day are single
    choice but Month is multi-select, so one request = one year x one init day x all 12 months,
    Forecast Day T..T+9: 23 x 4 = 92 requests."""
    variables = "Total Precipitation Amount, Mean Sea Level Pressure, Surface Pressure, 10 Metre U/V Wind Component"
    return pd.DataFrame([{"year": y, "init_day": d, "months": "all 12", "forecast_days": "T..T+9",
                          "variables": variables, "north": 38, "south": 6, "east": 98, "west": 68}
                         for y in years for d in init_days])


STATE_COLS = ["mslp", "surface_pressure", "wind_u10", "wind_v10"]


def build_pairs(fc, obs):
    """Pairs table from subdivision forecasts (to_subdivisions output) and IMD subdivision
    observations (columns date = IMD date, subdivision_code, observed_rain_mm).

    - observed_rain_mm: IMD on the valid date (issue date + lead_day).
    - Weather state (MSLP, surface pressure, 10 m winds): the earliest lead available (day01 on
      the portal), repeated for all 10 lead days as the issue-time state the feature builder expects.
    - total_precipitation: IMD rain on issue date - 1, the last IMD day that is complete
      before the 00Z issue (IMD day D ends 03Z on D). Known at issue time, so leak-free.
    Output uses the data team's column names for rain (forecast_rain_mm, ...)."""
    fc = fc.copy()
    fc["valid_date"] = fc["date"] + pd.to_timedelta(fc["lead_day"], "D")
    o = obs.rename(columns={"date": "valid_date"})
    pairs = fc.drop(columns=[c for c in STATE_COLS if c in fc]).merge(o, on=["valid_date", "subdivision_code"],
                                                                     how="left")
    state = [c for c in STATE_COLS if c in fc]
    if state:
        # Earliest lead each variable exists at. The portal has no "T" field for instantaneous
        # variables, so this is the model's day01 state (24 h after init): NCMRWF output that
        # exists at issue time, repeated for every lead as the issue-time state.
        s0 = (fc.sort_values("lead_day")[["date", "subdivision_code"] + state]
              .groupby(["date", "subdivision_code"], as_index=False).first())
        pairs = pairs.merge(s0, on=["date", "subdivision_code"], how="left")
    prev = obs.rename(columns={"observed_rain_mm": "total_precipitation"}).assign(
        date=lambda d: d["date"] + pd.Timedelta(days=1))
    pairs = pairs.merge(prev, on=["date", "subdivision_code"], how="left")
    pairs["error_mm"] = pairs["forecast_rain_mm"] - pairs["observed_rain_mm"]
    pairs["year"] = pairs["date"].dt.year
    cols = ["date", "subdivision_code", "lead_day", "forecast_rain_mm", "observed_rain_mm", "error_mm"] + state + \
        ["total_precipitation", "year"]
    return pairs[cols].sort_values(["date", "subdivision_code", "lead_day"]).reset_index(drop=True)
