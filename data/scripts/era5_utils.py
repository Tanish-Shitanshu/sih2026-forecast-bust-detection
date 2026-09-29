"""
Loads one year's ERA5 data and aggregates down to subdivision-level daily
values. Two sources, both needed:

- era5_{year}.nc (downloaded as .nc but actually a zip of two NetCDFs: an
  "instant" file with u10/v10/d2m/t2m/msl/sp at 4 synoptic hours/day, and
  an "accum" file with total_precipitation at those same 4 hours) ->
  era5_subdivision_daily() gives the daily MEAN of the 4 instantaneous
  fields. It intentionally does NOT return precipitation: CDS delivers
  ERA5 tp as clean 1-hour increments, so summing only 4 of a day's 24
  hourly values recovers roughly 4/24 (~17%) of the real daily total --
  confirmed empirically (0.185x IMD's same-day total across 2015). That
  bug shipped once; the fix is to not expose the wrong quantity from this
  function at all, not just to remember not to call it.
- data/raw/era5_hourly_tp/era5_tp_hourly_{year}.nc (all 24 hourly steps,
  downloaded separately via download_era5_hourly_tp.py) ->
  hourly_tp_subdivision_daily() sums all 24 values for a true daily total.

LEAKAGE WARNING, read before using either function's return value: their
keys are the calendar day the ERA5 observation is FROM, not the day a
forecast about it may be issued. A GEFS forecast issued at 00Z on day D can
only ever see weather-state data through day D-1 -- three of day D's own
four synoptic snapshots (06/12/18Z), and effectively all 24 of its hourly
precipitation values, are strictly in that forecast's future. Do not index
either dict by a forecast's init date directly; use
`era5_features_for_issue_date` below, which applies the -1 day shift so
that can't be forgotten at a call site.
"""
import os
import datetime
import zipfile
import numpy as np
import xarray as xr

ERA5_DIR = "data/raw/era5"
EXTRACT_DIR = os.path.join(ERA5_DIR, "_extracted")
HOURLY_TP_DIR = "data/raw/era5_hourly_tp"

INSTANT_FEATURES = ["msl", "sp", "d2m", "t2m", "u10", "v10"]


def _extract_member(year: int, member_substr: str, out_name: str) -> str:
    out_dir = os.path.join(EXTRACT_DIR, str(year))
    out_path = os.path.join(out_dir, out_name)
    if os.path.exists(out_path):
        return out_path
    os.makedirs(out_dir, exist_ok=True)
    zip_path = os.path.join(ERA5_DIR, f"era5_{year}.nc")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        member_name = next(n for n in names if member_substr in n)
        zf.extract(member_name, out_dir)
        os.replace(os.path.join(out_dir, member_name), out_path)
    return out_path


def _extract_year(year: int) -> str:
    _extract_member(year, "instant", "instant.nc")
    return os.path.join(EXTRACT_DIR, str(year))


def _code_to_ij(grid_map, cache):
    if cache is not None:
        return cache
    out = {}
    for (i, j), code in grid_map.items():
        out.setdefault(code, []).append((i, j))
    return out


def era5_subdivision_daily(year: int, grid_map, code_to_ij_cache=None):
    """
    Returns dict {(date: python date, subdivision_code): {feature: value}}
    for msl, sp, d2m, t2m, u10, v10 -- the daily mean of the 4 synoptic
    hours. Precipitation is NOT included here; see hourly_tp_subdivision_daily.
    """
    code_to_ij = _code_to_ij(grid_map, code_to_ij_cache)
    result = {}

    instant_path = _extract_member(year, "instant", "instant.nc")
    ds = xr.open_dataset(instant_path)
    dates = ds.valid_time.dt.floor("D").values
    unique_dates = np.unique(dates)
    for feat in INSTANT_FEATURES:
        arr = ds[feat].values  # (valid_time, lat, lon)
        for code, ij_list in code_to_ij.items():
            ii = np.array([p[0] for p in ij_list])
            jj = np.array([p[1] for p in ij_list])
            cell_series = arr[:, ii, jj]  # (valid_time, n_cells)
            spatial_mean = np.nanmean(cell_series, axis=1)  # (valid_time,)
            for date in unique_dates:
                day_mask = dates == date
                day_mean = float(np.nanmean(spatial_mean[day_mask]))
                date_key = np.datetime64(date, "D").astype(object)  # plain python date
                result.setdefault((date_key, code), {})[feat] = day_mean
    ds.close()

    return result


def hourly_tp_subdivision_daily(year: int, grid_map, code_to_ij_cache=None):
    """
    Returns dict {(date: python date, subdivision_code): tp_mm} -- a true
    daily total precipitation, summed from all 24 hourly ERA5 values (each
    a clean 1-hour increment) and converted m -> mm. Requires
    data/raw/era5_hourly_tp/era5_tp_hourly_{year}.nc, from
    download_era5_hourly_tp.py.
    """
    code_to_ij = _code_to_ij(grid_map, code_to_ij_cache)
    path = os.path.join(HOURLY_TP_DIR, f"era5_tp_hourly_{year}.nc")
    ds = xr.open_dataset(path)
    dates = ds.valid_time.dt.floor("D").values
    unique_dates = np.unique(dates)
    arr = ds["tp"].values * 1000.0  # meters -> mm
    result = {}
    for code, ij_list in code_to_ij.items():
        ii = np.array([p[0] for p in ij_list])
        jj = np.array([p[1] for p in ij_list])
        cell_series = arr[:, ii, jj]
        spatial_mean = np.nanmean(cell_series, axis=1)
        for date in unique_dates:
            day_mask = dates == date
            n_hours = int(day_mask.sum())
            day_sum = float(np.nansum(spatial_mean[day_mask]))
            if n_hours != 24:
                continue  # partial day (shouldn't happen mid-year, guards year boundaries)
            date_key = np.datetime64(date, "D").astype(object)
            result[(date_key, code)] = day_sum
    ds.close()
    return result


def merge_tp_into_daily(era5_daily, tp_daily):
    """Merges hourly_tp_subdivision_daily's output into era5_subdivision_daily's,
    under the "tp" key, in place. Returns era5_daily for convenience."""
    for key, tp_mm in tp_daily.items():
        era5_daily.setdefault(key, {})["tp"] = tp_mm
    return era5_daily


def era5_features_for_issue_date(era5_daily, issue_date, code):
    """
    The only sanctioned way to attach ERA5 features to a forecast row.
    Returns era5_daily's aggregate for issue_date - 1 day: the most recent
    full calendar day that was entirely in the past when a forecast issued
    on issue_date (00Z) was made. Returns {} if that day isn't in
    era5_daily (e.g. it needs the previous year's file and that wasn't
    loaded/merged in).
    """
    lookup_date = issue_date - datetime.timedelta(days=1)
    return era5_daily.get((lookup_date, code), {})


def build_era5_grid_map():
    from grid_utils import load_boundaries, build_grid_subdivision_map
    out_dir = _extract_year(2015)  # any year — same grid every year
    ds = xr.open_dataset(os.path.join(out_dir, "instant.nc"))
    lats = ds.latitude.values
    lons = ds.longitude.values
    ds.close()
    geoms, codes = load_boundaries()
    return build_grid_subdivision_map(lats, lons, geoms, codes)
