"""
Loads one year's ERA5 file (downloaded as era5_{year}.nc, which despite the
extension is actually a zip of two NetCDFs — one "instant" file with
u10/v10/d2m/t2m/msl/sp, one "accum" file with tp) and aggregates the 4
synoptic hours (00/06/12/18Z) of each day down to a subdivision-level daily
mean for a few basic pressure/moisture features.
"""
import os
import zipfile
import numpy as np
import xarray as xr

ERA5_DIR = "data/raw/era5"
EXTRACT_DIR = os.path.join(ERA5_DIR, "_extracted")

FEATURES = ["msl", "sp", "d2m", "t2m"]  # pressure (msl, sp) + moisture/temp proxy (d2m, t2m)


def _extract_year(year: int) -> str:
    out_dir = os.path.join(EXTRACT_DIR, str(year))
    instant_path = os.path.join(out_dir, "instant.nc")
    if os.path.exists(instant_path):
        return out_dir
    os.makedirs(out_dir, exist_ok=True)
    zip_path = os.path.join(ERA5_DIR, f"era5_{year}.nc")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        instant_name = next(n for n in names if "instant" in n)
        zf.extract(instant_name, out_dir)
        os.replace(os.path.join(out_dir, instant_name), instant_path)
    return out_dir


def era5_subdivision_daily(year: int, grid_map, code_to_ij_cache=None):
    """
    Returns dict {(date: numpy.datetime64[D], subdivision_code): {feature: value}}
    with the 4 synoptic hours and all grid cells in a subdivision averaged
    together per day.
    """
    out_dir = _extract_year(year)
    ds = xr.open_dataset(os.path.join(out_dir, "instant.nc"))

    code_to_ij = code_to_ij_cache
    if code_to_ij is None:
        code_to_ij = {}
        for (i, j), code in grid_map.items():
            code_to_ij.setdefault(code, []).append((i, j))

    dates = ds.valid_time.dt.floor("D").values
    unique_dates = np.unique(dates)

    result = {}
    for feat in FEATURES:
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


def build_era5_grid_map():
    from grid_utils import load_boundaries, build_grid_subdivision_map
    out_dir = _extract_year(2015)  # any year — same grid every year
    ds = xr.open_dataset(os.path.join(out_dir, "instant.nc"))
    lats = ds.latitude.values
    lons = ds.longitude.values
    ds.close()
    geoms, codes = load_boundaries()
    return build_grid_subdivision_map(lats, lons, geoms, codes)
