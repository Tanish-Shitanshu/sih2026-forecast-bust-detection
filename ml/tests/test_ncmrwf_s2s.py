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
    assert len(p) == 23 * 12 and (p["east"] > p["west"]).all()
