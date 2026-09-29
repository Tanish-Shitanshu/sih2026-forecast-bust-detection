"""NCMRWF S2S reader, on small files shaped exactly like the portal's download."""
import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("netCDF4")

from vishwas_ml.ncmrwf_s2s import read_request, request_plan, to_subdivisions  # noqa: E402


def _write(folder, lon0, lon1, init="20150701", days=11, scale=1.0):
    lat = np.arange(6.389, 38.0, 0.5555556, dtype="float32")
    lon = np.arange(lon0, lon1, 0.8333333, dtype="float32")
    for nn in range(days):
        a = np.full((1, lat.size, lon.size), 5.0 + nn, "float32") * scale
        ds = xr.Dataset({"apcp": (("t", "latitude", "longitude"), a, {"units": "kg m-2 s-1"})},
                        coords={"t": [pd.Timestamp(init) + pd.Timedelta(days=nn)], "latitude": lat, "longitude": lon})
        ds.to_netcdf(folder / f"APCP-sfc_IC{init}_day{nn:02d}.nc")


def test_reads_india_box_and_maps_lead_days(tmp_path):
    _write(tmp_path, 68.33, 98.0)
    g = read_request(tmp_path)
    assert sorted(g["lead_day"].unique()) == list(range(1, 11))       # day00..day09 -> lead 1..10
    assert g["init_date"].eq(pd.Timestamp("2015-07-01")).all()
    assert np.allclose(g.loc[g["lead_day"] == 1, "forecast_rain_mm"], 5.0)   # day00 -> lead 1
    assert g["longitude"].between(68, 98).all() and g["latitude"].between(6, 38).all()


def test_rejects_swapped_east_west(tmp_path):
    _write(tmp_path, 90.42, 428.0)  # what the portal returns for EAST=68, WEST=90
    with pytest.raises(ValueError, match="swapped"):
        read_request(tmp_path)


def test_rejects_implausible_units(tmp_path):
    _write(tmp_path, 68.33, 98.0, scale=86400.0)
    with pytest.raises(ValueError, match="units"):
        read_request(tmp_path)


def test_subdivision_weighting(tmp_path):
    _write(tmp_path, 68.33, 98.0, days=2)
    g = read_request(tmp_path, max_day=0)
    cells = g[["latitude", "longitude"]].drop_duplicates().head(4)
    w = cells.assign(subdivision_code=["KL", "KL", "TN/PY", "TN/PY"], weight=[1.0, 3.0, 1.0, 1.0])
    g.loc[(g["latitude"] == cells.iloc[1]["latitude"]) & (g["longitude"] == cells.iloc[1]["longitude"]),
          "forecast_rain_mm"] = 9.0
    out = to_subdivisions(g, w).set_index("subdivision_code")["forecast_rain_mm"]
    assert np.isclose(out["KL"], (5.0 * 1 + 9.0 * 3) / 4) and np.isclose(out["TN/PY"], 5.0)


def test_request_plan_covers_all_months():
    p = request_plan()
    assert len(p) == 23 * 4 and (p["east"] > p["west"]).all()


def test_float32_grid_joins_float64_weights(tmp_path):
    """NetCDF coordinates are float32, weight tables float64 (CSV): every cell must still match."""
    _write(tmp_path, 68.33, 98.0, days=1)
    g = read_request(tmp_path, max_day=0)
    cells = g[["latitude", "longitude"]].drop_duplicates()
    w = cells.astype("float32").astype("float64").round(6).assign(subdivision_code="X", weight=1.0)
    csv = tmp_path / "w.csv"
    w.to_csv(csv, index=False)
    out = to_subdivisions(g.astype({"latitude": "float32", "longitude": "float32"}), pd.read_csv(csv))
    assert len(out) == 1 and np.isclose(out["forecast_rain_mm"].iloc[0], 5.0)


def _write_var(folder, fname_var, name, std, value, init="20150701", days=3):
    lat = np.arange(6.389, 38.0, 0.5555556, dtype="float32")
    lon = np.arange(68.33, 98.0, 0.8333333, dtype="float32")
    for nn in range(days):
        a = np.full((1, lat.size, lon.size), value + nn, "float32")
        ds = xr.Dataset({name: (("t", "latitude", "longitude"), a, {"standard_name": std})},
                        coords={"t": [pd.Timestamp(init) + pd.Timedelta(days=nn)], "latitude": lat, "longitude": lon})
        ds.to_netcdf(folder / f"{fname_var}_IC{init}_day{nn:02d}.nc")


def test_reads_weather_variables_by_metadata(tmp_path):
    _write(tmp_path, 68.33, 98.0, days=3)
    _write_var(tmp_path, "PRMSL-msl", "prmsl", "air_pressure_at_sea_level", 100500.0)
    _write_var(tmp_path, "UGRD-10m", "ugrd", "eastward_wind", 3.0)
    _write_var(tmp_path, "MYSTERY-x", "zzz", "unknown_thing", 1.0)
    with pytest.warns(UserWarning, match="unrecognised"):
        g = read_request(tmp_path, max_day=2)
    assert {"forecast_rain_mm", "mslp", "wind_u10"} <= set(g.columns)
    assert np.allclose(g.loc[g["lead_day"] == 1, "mslp"], 100500.0)


def test_build_pairs_timing():
    from vishwas_ml.ncmrwf_s2s import build_pairs

    d0 = pd.Timestamp("2015-07-09")
    fc = pd.DataFrame({"date": d0, "subdivision_code": "KL", "lead_day": [1, 2, 3],
                       "forecast_rain_mm": [10.0, 20.0, 30.0], "mslp": [1000.0, 990.0, 980.0]})
    obs = pd.DataFrame({"date": pd.date_range("2015-07-07", periods=6), "subdivision_code": "KL",
                        "observed_rain_mm": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})  # 7th..12th July
    p = build_pairs(fc, obs)
    assert p["observed_rain_mm"].tolist() == [4.0, 5.0, 6.0]      # valid = issue + lead
    assert (p["mslp"] == 1000.0).all()                             # issue-time state for every lead
    assert (p["total_precipitation"] == 2.0).all()                 # IMD issue date - 1 (8 July), never the 9th
    assert np.allclose(p["error_mm"], [6.0, 15.0, 24.0])


def test_staggered_wind_grid_is_averaged_on_its_own_cells(tmp_path):
    """UM winds sit half a cell east-west of rain/pressure; each must use its own weights."""
    _write(tmp_path, 68.75, 98.0, days=2)                       # rain grid
    lat = np.arange(6.389, 38.0, 0.5555556, dtype="float32")
    lon_uv = np.arange(68.333, 97.6, 0.8333333, dtype="float32")  # staggered grid
    a = np.full((1, lat.size, lon_uv.size), 4.0, "float32")
    xr.Dataset({"u": (("t", "latitude", "longitude"), a, {"standard_name": "eastward_wind"})},
               coords={"t": [pd.Timestamp("2015-07-02")], "latitude": lat, "longitude": lon_uv}
               ).to_netcdf(tmp_path / "UGRD-10m_IC20150701_day01.nc")
    g = read_request(tmp_path, max_day=1)
    cells = lambda c: g.loc[g[c].notna(), ["latitude", "longitude"]].drop_duplicates()  # noqa: E731
    w = pd.concat([cells("forecast_rain_mm").assign(subdivision_code="X", weight=1.0),
                   cells("wind_u10").assign(subdivision_code="X", weight=1.0)])
    out = to_subdivisions(g, w)
    assert np.isclose(out.loc[out["lead_day"] == 2, "wind_u10"].iloc[0], 4.0)
    assert np.isclose(out.loc[out["lead_day"] == 1, "forecast_rain_mm"].iloc[0], 5.0)
