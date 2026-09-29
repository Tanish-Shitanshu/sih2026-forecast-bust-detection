import sys
from pathlib import Path

import pytest

ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML))

from vishwas_ml.config import load_config  # noqa: E402
from vishwas_ml.synthetic import generate  # noqa: E402


@pytest.fixture(scope="session")
def small_data():
    """Small synthetic table (issue every 9 days) so the suite trains in seconds."""
    return generate(seed=1, every=9)


@pytest.fixture(scope="session")
def bundle(small_data, tmp_path_factory):
    from vishwas_ml.pipeline import train_pipeline

    d = tmp_path_factory.mktemp("bundle")
    data = d / "pairs.parquet"
    small_data.to_parquet(data, index=False)
    cfg = load_config(source="synthetic")
    cfg["data_path"], cfg["models_dir"] = data, d / "models"
    cfg["lightgbm"] = dict(cfg["lightgbm"], n_estimators=300)
    res = train_pipeline(cfg, log=lambda *a: None)
    return {"cfg": cfg, "dir": d / "models", "data": small_data, "result": res}


@pytest.fixture(scope="session")
def service(bundle, tmp_path_factory):
    from vishwas_ml.service import VishwasService

    fb = tmp_path_factory.mktemp("fb") / "feedback.jsonl"
    return VishwasService(models_dir=bundle["dir"], history=bundle["data"], cfg=bundle["cfg"], feedback_path=fb)
