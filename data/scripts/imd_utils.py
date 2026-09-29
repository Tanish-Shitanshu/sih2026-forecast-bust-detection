"""Loads IMD gridded rainfall for one year and aggregates to subdivision daily means."""
import os
import numpy as np
import xarray as xr
from grid_utils import load_boundaries, build_grid_subdivision_map

IMD_DIR = "data/raw/imd"


def load_imd_year(year: int):
    """Returns an xarray.DataArray RAINFALL(TIME, LATITUDE, LONGITUDE) for one year."""
    path = os.path.join(IMD_DIR, f"ind{year}_rfp25.nc")
    ds = xr.open_dataset(path)
    return ds.RAINFALL


def imd_subdivision_daily(year: int, grid_map=None, geoms=None, codes=None):
    """
    Returns a dict {(date: numpy.datetime64, subdivision_code): observed_rain_mm}
    by averaging every grid cell assigned to a subdivision, for every day in
    the given year's IMD file.
    """
    rainfall = load_imd_year(year)
    lats = rainfall.LATITUDE.values
    lons = rainfall.LONGITUDE.values

    if grid_map is None:
        if geoms is None or codes is None:
            geoms, codes = load_boundaries()
        grid_map = build_grid_subdivision_map(lats, lons, geoms, codes)

    # group grid indices by subdivision code once
    code_to_ij = {}
    for (i, j), code in grid_map.items():
        code_to_ij.setdefault(code, []).append((i, j))

    values = rainfall.values  # (TIME, LAT, LON)
    # normalize to plain python date objects so keys match across modules
    times = [np.datetime64(t, "D").astype(object) for t in rainfall.TIME.values]

    out = {}
    for code, ij_list in code_to_ij.items():
        ii = np.array([p[0] for p in ij_list])
        jj = np.array([p[1] for p in ij_list])
        cell_series = values[:, ii, jj]  # (TIME, n_cells)
        daily_mean = np.nanmean(cell_series, axis=1)  # (TIME,)
        for t_idx, date in enumerate(times):
            out[(date, code)] = float(daily_mean[t_idx])
    return out, grid_map
