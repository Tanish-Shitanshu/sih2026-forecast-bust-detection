import json
import re

import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

from vishwas_ml.config import ML_DIR
from vishwas_ml.model import BustModel, Isotonic


def test_isotonic_matches_sklearn_and_is_monotone():
    rng = np.random.default_rng(0)
    p = rng.random(3000)
    y = (rng.random(3000) < p ** 2).astype(float)
    ours = Isotonic().fit(p, y)
    sk = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p, y)
    q = np.linspace(-0.1, 1.1, 500)
    assert np.allclose(ours(q), sk.predict(q))
    assert np.all(np.diff(ours(q)) >= 0)
    assert np.allclose(Isotonic.from_dict(json.loads(json.dumps(ours.to_dict())))(q), ours(q))


def test_contributions_are_exact_shap(bundle):
    shap = pytest.importorskip("shap")
    m = BustModel.load(bundle["dir"])
    X = m.builder.transform(bundle["data"]).iloc[:200]
    ours = m.contributions(X)
    ref = shap.TreeExplainer(m.booster).shap_values(X)
    ref = ref[1] if isinstance(ref, list) else ref
    assert np.allclose(ours, ref, atol=1e-5)
    # SHAP values sum to the raw log-odds.
    raw = m.predict_raw(X)
    base = m.booster.predict(X, pred_contrib=True)[:, -1]
    assert np.allclose(ours.sum(1) + base, np.log(raw / (1 - raw)), atol=1e-5)


def test_bundle_roundtrip_predicts_the_same(bundle, tmp_path):
    m = BustModel.load(bundle["dir"])
    X = m.builder.transform(bundle["data"]).iloc[:500]
    m.save(tmp_path)
    m2 = BustModel.load(tmp_path)
    assert np.allclose(m.predict(X), m2.predict(X))
    assert np.allclose(m2.builder.transform(bundle["data"]).iloc[:500], X, equal_nan=True)


def test_metrics_beat_baselines_and_are_calibrated(bundle):
    ov = bundle["result"]["metrics"]["overall"]
    assert 0.10 <= ov["model"]["base_rate"] <= 0.25
    assert ov["model"]["pr_auc"] > ov["baseline_climatology"]["pr_auc"] > ov["model"]["base_rate"]
    assert ov["model"]["ece"] < 0.03


def test_recalibration_needs_enough_approved_outcomes(bundle):
    from vishwas_ml.feedback import recalibrate

    m = BustModel.load(bundle["dir"])
    cal = pd.read_parquet(bundle["dir"] / "calibration_set.parquet")
    fb = pd.DataFrame({"predicted_bust_probability": [0.7] * 10, "outcome": ["incorrect"] * 10,
                       "status": ["approved"] * 10})
    rc, s = recalibrate(m, fb, cal, {"min_outcomes": 50, "weight": 1.0})
    assert rc is None and s["status"].startswith("skipped")
    fb = pd.DataFrame({"predicted_bust_probability": np.r_[np.full(60, 0.2), np.full(60, 0.8)],
                       "outcome": ["incorrect"] * 60 + ["correct"] * 60, "status": ["approved"] * 120})
    fb.loc[:5, "status"] = "pending"
    rc, s = recalibrate(m, fb, cal, {"min_outcomes": 50, "weight": 50.0})
    assert rc is not None and s["approved_used"] == 114
    assert rc(np.array([0.2]))[0] > 0.2  # feedback says low-risk forecasts busted


def test_subdivisions_match_frontend():
    import sys
    sys.path.insert(0, str(ML_DIR / "scripts"))
    from sync_subdivisions import FRONTEND, parse

    fresh = parse(FRONTEND.read_text(encoding="utf-8"))
    saved = json.loads((ML_DIR / "data" / "subdivisions.json").read_text(encoding="utf-8"))
    assert fresh["subdivisions"] == saved["subdivisions"], "run ml/scripts/sync_subdivisions.py"
    assert fresh["tiers"] == saved["tiers"] and fresh["trust_levels"] == saved["trust_levels"]
    assert len(saved["subdivisions"]) == 33
    assert all(re.fullmatch(r"[A-Z&./]+", s["code"]) for s in saved["subdivisions"])


def test_schema_validation_rejects_bad_tables(small_data):
    from vishwas_ml.config import subdivision_codes
    from vishwas_ml.schema import SchemaError, validate

    codes = subdivision_codes()
    ok = small_data.head(3300)
    assert validate(ok, codes)["era5_timing_detected"] == "issue"
    for bad in (ok.assign(subdivision_code="NOPE"), ok.assign(lead_day=11), pd.concat([ok, ok.head(1)])):
        with pytest.raises(SchemaError):
            validate(bad, codes)
    with pytest.raises(SchemaError):
        validate(ok.drop(columns="forecast_rain"), codes)
    validate(ok.drop(columns="mslp"), codes)  # ERA5 variables are optional
