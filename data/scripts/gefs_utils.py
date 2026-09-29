"""
Decodes one GEFS reforecast apcp_sfc GRIB2 file (control member, all Day 1-10
lead times, global 0.25 deg) into daily (24h) subdivision-mean precipitation
totals for lead days 1-10.
"""
import os
import numpy as np
import xarray as xr

GEFS_DIR = "data/raw/gefs"


def gefs_daily_subdivision(date_str_yyyymmdd: str, grid_map, code_to_ij_cache=None):
    """
    date_str_yyyymmdd: e.g. "20150601"
    grid_map: dict (i,j)->subdivision_code, built once against this GEFS file's
              own lat/lon grid (see build_gefs_grid_map below).
    Returns dict {(lead_day 1-10, subdivision_code): forecast_rain_mm}, or
    None if the file is missing.
    """
    path = os.path.join(GEFS_DIR, f"apcp_sfc_{date_str_yyyymmdd}00_c00.grib2")
    if not os.path.exists(path):
        return None

    ds = xr.open_dataset(path, engine="cfgrib",
                          filter_by_keys={"stepType": "accum"},
                          backend_kwargs={"indexpath": ""})
    hours = ds.step.values / np.timedelta64(1, "h")
    six_hourly = ds.sel(step=ds.step[hours % 6 == 0])  # 40 clean, non-overlapping 6h blocks
    tp = six_hourly.tp.values  # (40, lat, lon)
    daily = tp.reshape(10, 4, tp.shape[1], tp.shape[2]).sum(axis=1)  # (10 days, lat, lon)

    code_to_ij = code_to_ij_cache
    if code_to_ij is None:
        code_to_ij = {}
        for (i, j), code in grid_map.items():
            code_to_ij.setdefault(code, []).append((i, j))

    out = {}
    for code, ij_list in code_to_ij.items():
        ii = np.array([p[0] for p in ij_list])
        jj = np.array([p[1] for p in ij_list])
        for lead_day in range(1, 11):
            cells = daily[lead_day - 1, ii, jj]
            out[(lead_day, code)] = float(np.nanmean(cells))
    ds.close()
    return out


def build_gefs_grid_map():
    """
    Builds the (i,j)->subdivision_code map for GEFS's own global 0.25-deg
    grid, using any one already-downloaded GEFS file to read the exact
    lat/lon coordinates (GEFS uses 0-360 longitude; grid_utils normalizes).
    """
    from grid_utils import load_boundaries, build_grid_subdivision_map

    sample = sorted(f for f in os.listdir(GEFS_DIR) if f.endswith(".grib2"))
    if not sample:
        raise RuntimeError("No GEFS files downloaded yet — run download_gefs.py first")
    path = os.path.join(GEFS_DIR, sample[0])
    ds = xr.open_dataset(path, engine="cfgrib",
                          filter_by_keys={"stepType": "accum"},
                          backend_kwargs={"indexpath": ""})
    lats = ds.latitude.values
    lons = ds.longitude.values
    ds.close()
    geoms, codes = load_boundaries()
    return build_grid_subdivision_map(lats, lons, geoms, codes)
