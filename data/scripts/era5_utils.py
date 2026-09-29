"""
Loads one year's ERA5 file (downloaded as era5_{year}.nc, which despite the
extension is actually a zip of two NetCDFs — one "instant" file with
u10/v10/d2m/t2m/msl/sp, one "accum" file with tp) and aggregates the 4
synoptic hours (00/06/12/18Z) of each day down to a subdivision-level daily
value: a mean for the instantaneous fields, a sum (the 4 non-overlapping
6h buckets) for accumulated total precipitation, converted m -> mm.
"""
import os
import zipfile
import numpy as np
import xarray as xr

ERA5_DIR = "data/raw/era5"
EXTRACT_DIR = os.path.join(ERA5_DIR, "_extracted")

INSTANT_FEATURES = ["msl", "sp", "d2m", "t2m", "u10", "v10"]
ACCUM_FEATURE = "tp"  # meters -> mm


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
    Returns dict {(date: python date, subdivision_code): {feature: value}}:
    msl, sp, d2m, t2m, u10, v10 as the daily mean of the 4 synoptic hours;
    tp as the daily SUM of the 4 (non-overlapping) 6h accumulations, in mm.
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

    accum_path = _extract_member(year, "accum", "accum.nc")
    ds = xr.open_dataset(accum_path)
    dates = ds.valid_time.dt.floor("D").values
    unique_dates = np.unique(dates)
    arr = ds[ACCUM_FEATURE].values * 1000.0  # meters -> mm
    for code, ij_list in code_to_ij.items():
        ii = np.array([p[0] for p in ij_list])
        jj = np.array([p[1] for p in ij_list])
        cell_series = arr[:, ii, jj]
        spatial_mean = np.nanmean(cell_series, axis=1)
        for date in unique_dates:
            day_mask = dates == date
            day_sum = float(np.nansum(spatial_mean[day_mask]))
            date_key = np.datetime64(date, "D").astype(object)
            result.setdefault((date_key, code), {})[ACCUM_FEATURE] = day_sum
    ds.close()

    return result


def build_era5_grid_map():
    from grid_utils import load_boundaries, build_grid_subdivision_map
    out_dir = _extract_year(2015)  # any year — same grid every year
    ds = xr.open_dataset(os.path.join(out_dir, "instant.nc"))
    lats = ds.latitude.values
    lons = ds.longitude.values
    ds.close()
    geoms, codes = load_boundaries()
    return build_grid_subdivision_map(lats, lons, geoms, codes)
