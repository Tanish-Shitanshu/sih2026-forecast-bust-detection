"""Reader for NCMRWF's operational ensemble (NEPS) forecasts from the TIGGE archive
(ECMWF Data Store, dataset "tigge-forecasts", origin "ncmrwf"; available 2017-2025,
00Z/12Z, lead hours 0-240).

A request returns GRIB2. Per run (init date, 00Z) this module produces:
  * daily rain for lead days 1..10: TIGGE precipitation is accumulated from step 0, so
    lead day L = tp(24 L h) - tp(24 (L-1) h), i.e. the 24 h ending 00Z on init+L. That pairs
    with IMD's day init+L (24 h ending 03Z), the same convention as the S2S reader (lead_day L,
    valid date = issue date + L).
  * the state at issue time from step 0 (MSLP, surface pressure, 2 m temperature and
    dewpoint, 10 m winds): the model's own analysis at 00Z, known when the forecast is issued.
Output rows use lead_day 0 for the issue-time state and 1..10 for rain, so the existing
subdivision aggregation (ncmrwf_s2s.to_subdivisions) and pairing can be reused.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

INDIA = {"lat": (6.0, 38.0), "lon": (68.0, 98.0)}
# GRIB shortName -> output column
STATE = {"msl": "mslp", "sp": "surface_pressure", "2t": "temp_2m", "t2m": "temp_2m", "2d": "dewpoint_2m",
         "d2m": "dewpoint_2m", "10u": "wind_u10", "u10": "wind_u10", "10v": "wind_v10", "v10": "wind_v10"}


def _open(path):
    """All GRIB messages as a list of xarray datasets (cfgrib splits incompatible hypercubes)."""
    import cfgrib
    return cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""})


def _tidy(da, col, init, lead):
    lat = da["latitude"].values.astype("float64")
    lon = ((da["longitude"].values.astype("float64") + 180) % 360) - 180
    klat = (lat >= INDIA["lat"][0]) & (lat <= INDIA["lat"][1])
    klon = (lon >= INDIA["lon"][0]) & (lon <= INDIA["lon"][1])
    v = np.asarray(da.values, dtype="float64")[np.ix_(klat, klon)]
    LA, LO = np.meshgrid(lat[klat], lon[klon], indexing="ij")
    return pd.DataFrame({"init_date": init, "lead_day": lead, "latitude": LA.ravel(), "longitude": LO.ravel(),
                         col: v.ravel()})


def read_request(path, member="control"):
    """Tidy wide frame: init_date, lead_day (0 = issue-time state, 1..10 = daily rain), latitude,
    longitude and the value columns. member: "control" or "mean" (mean over perturbed members)."""
    parts = {}
    for ds in _open(path):
        for name in ds.data_vars:
            da = ds[name]
            short = str(da.attrs.get("GRIB_shortName", name)).lower()
            if "number" in da.dims:
                da = da.mean("number") if member == "mean" else da.isel(number=0)
            times = np.atleast_1d(da["time"].values)
            steps = np.atleast_1d(da["step"].values) if "step" in da.coords else np.array([np.timedelta64(0, "h")])
            for ti, t in enumerate(times):
                init = pd.Timestamp(t).normalize()
                dt = da.isel(time=ti) if "time" in da.dims else da
                hours = (pd.to_timedelta(steps) / pd.Timedelta(hours=1)).astype(int)
                if short == "tp":
                    acc = {h: (dt.isel(step=i) if "step" in dt.dims else dt) for i, h in enumerate(hours)}
                    for L in range(1, 11):
                        if 24 * L in acc and 24 * (L - 1) in acc:
                            d = acc[24 * L] - acc[24 * (L - 1)]
                            parts.setdefault("forecast_rain_mm", []).append(
                                _tidy(d.clip(min=0), "forecast_rain_mm", init, L))
                elif short in STATE:
                    if 0 not in list(hours):
                        continue
                    i0 = list(hours).index(0)
                    d = dt.isel(step=i0) if "step" in dt.dims else dt
                    parts.setdefault(STATE[short], []).append(_tidy(d, STATE[short], init, 0))
                else:
                    warnings.warn(f"{Path(path).name}: unrecognised GRIB field {short!r} skipped")
    if not parts:
        raise ValueError(f"no usable fields in {path}")
    keys = ["init_date", "lead_day", "latitude", "longitude"]
    out = None
    for col, dfs in parts.items():
        d = pd.concat(dfs, ignore_index=True).drop_duplicates(keys)
        out = d if out is None else out.merge(d, on=keys, how="outer")
    rain = out["forecast_rain_mm"].dropna() if "forecast_rain_mm" in out else pd.Series(dtype=float)
    if len(rain) and not (0 <= rain.max() <= 2000):
        raise ValueError(f"implausible daily rain max {rain.max():.3g} mm; check units")
    return out.sort_values(keys, kind="stable").reset_index(drop=True)


def build_pairs(fc, obs):
    """Pairs table from subdivision-level TIGGE output (to_subdivisions of read_request) and IMD
    subdivision obs (date = IMD date, subdivision_code, observed_rain_mm). Issue-time state comes
    from lead_day 0 and is repeated for every lead; total_precipitation = IMD on issue date - 1."""
    state = [c for c in ("mslp", "surface_pressure", "temp_2m", "dewpoint_2m", "wind_u10", "wind_v10") if c in fc]
    rain = fc[fc["lead_day"] >= 1][["date", "subdivision_code", "lead_day", "forecast_rain_mm"]]
    s0 = fc[fc["lead_day"] == 0][["date", "subdivision_code"] + state]
    p = rain.assign(valid_date=rain["date"] + pd.to_timedelta(rain["lead_day"], "D"))
    p = p.merge(obs.rename(columns={"date": "valid_date"}), on=["valid_date", "subdivision_code"], how="left")
    p = p.merge(s0, on=["date", "subdivision_code"], how="left")
    prev = obs.rename(columns={"observed_rain_mm": "total_precipitation"}).assign(
        date=lambda d: d["date"] + pd.Timedelta(days=1))
    p = p.merge(prev, on=["date", "subdivision_code"], how="left")
    p["error_mm"] = p["forecast_rain_mm"] - p["observed_rain_mm"]
    p["year"] = p["date"].dt.year
    cols = ["date", "subdivision_code", "lead_day", "forecast_rain_mm", "observed_rain_mm", "error_mm"] + state + \
        ["total_precipitation", "year"]
    return p[cols].sort_values(["date", "subdivision_code", "lead_day"]).reset_index(drop=True)
