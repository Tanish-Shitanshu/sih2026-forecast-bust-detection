"""The data/ workstream's actual table (DATA_PIPELINE_STATUS.md, section 3) must train as-is:
    date, subdivision_code, subdivision_name, lead_day, forecast_rain_mm, observed_rain_mm,
    error_mm, era5_msl_pa, era5_sp_pa, era5_d2m_k, era5_t2m_k, year
No wind, no ERA5 precipitation, no is_bust / trigger_reason."""
import numpy as np
import pytest

from vishwas_ml.config import load_config, load_meta
from vishwas_ml.schema import SchemaError, load_pairs
from vishwas_ml.synthetic import generate

DATA_TEAM = {"forecast_rain": "forecast_rain_mm", "observed_rain": "observed_rain_mm", "error": "error_mm",
             "mslp": "era5_msl_pa", "surface_pressure": "era5_sp_pa", "dewpoint_2m": "era5_d2m_k",
             "temp_2m": "era5_t2m_k"}


def _data_team_table(df):
    names = {s["code"]: s["name"] for s in load_meta()["subdivisions"]}
    out = df.drop(columns=["wind_u10", "wind_v10", "total_precipitation", "is_bust", "trigger_reason"])
    out = out.rename(columns=DATA_TEAM)
    out.insert(2, "subdivision_name", out["subdivision_code"].map(names))
    out["year"] = out["date"].dt.year
    return out


def _train(tmp_path, table):
    from vishwas_ml.pipeline import train_pipeline

    path = tmp_path / "pairs.parquet"
    table.to_parquet(path, index=False)
    cfg = load_config(source="synthetic")
    cfg["data_path"], cfg["models_dir"] = path, tmp_path / "models"
    cfg["lightgbm"] = dict(cfg["lightgbm"], n_estimators=300)
    return cfg, train_pipeline(cfg, log=lambda *a: None)


def test_loader_maps_data_team_columns(small_data, tmp_path):
    p = tmp_path / "t.parquet"
    _data_team_table(small_data).to_parquet(p, index=False)
    df = load_pairs(p)
    assert np.allclose(df["forecast_rain"], small_data["forecast_rain"])
    assert np.allclose(df["mslp"], small_data["mslp"])
    assert df["wind_u10"].isna().all() and df["is_bust"].isna().all()
    assert set(df.attrs["absent_columns"]) >= {"wind_u10", "total_precipitation", "is_bust"}


def test_missing_required_column_is_rejected(small_data, tmp_path):
    p = tmp_path / "t.parquet"
    _data_team_table(small_data).drop(columns="observed_rain_mm").to_parquet(p, index=False)
    with pytest.raises(SchemaError):
        load_pairs(p)


def test_data_team_format_trains_and_serves(small_data, tmp_path):
    from vishwas_ml.service import VishwasService

    cfg, res = _train(tmp_path, _data_team_table(small_data))
    m = res["metrics"]["overall"]
    assert m["model"]["pr_auc"] > m["baseline_climatology"]["pr_auc"]
    live = _data_team_table(small_data).drop(columns=["observed_rain_mm", "error_mm"])  # outcomes unknown
    svc = VishwasService(models_dir=cfg["models_dir"], history=live, cfg=cfg)
    r = svc.explain("KL", 3)
    assert len(r["factors"]) == 3 and all("nan" not in f["feature"].lower() for f in r["factors"])
    assert svc.confidence_map(1)["count"] == 33


def test_ensemble_spread_is_used_when_present(tmp_path):
    cfg, res = _train(tmp_path, generate(seed=2, every=9, ensemble=True))
    imp = res["importance"]["groups"]
    assert imp["ensemble"] > 0
