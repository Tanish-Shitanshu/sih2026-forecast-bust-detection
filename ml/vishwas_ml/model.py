"""LightGBM bust classifier + isotonic calibration, and the saved model bundle.

Bundle layout (ml/models/):
    booster.txt          LightGBM model (text format)
    calibrator.json      isotonic map raw probability -> calibrated probability
    recalibrator.json    optional second isotonic map fit from approved forecaster outcomes
    features.json        fitted FeatureBuilder tables (climatology, thresholds, bust rates)
    analogs.parquet      analog library (standardised vectors + outcome metadata)
    analogs.json         analog index parameters
    metadata.json        data source, split, metrics summary, feature importance
"""
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression

from .features import FEATURES, FeatureBuilder


def train_booster(X_tr, y_tr, X_va, y_va, params):
    p = dict(objective="binary", metric="binary_logloss", verbose=-1, num_threads=0,
             num_leaves=params["num_leaves"], learning_rate=params["learning_rate"],
             min_child_samples=params["min_child_samples"], feature_fraction=params["feature_fraction"],
             bagging_fraction=params["bagging_fraction"], bagging_freq=params["bagging_freq"],
             lambda_l2=params["lambda_l2"], seed=params["seed"])
    dtr = lgb.Dataset(X_tr, np.asarray(y_tr, float), feature_name=list(X_tr.columns))
    dva = lgb.Dataset(X_va, np.asarray(y_va, float), reference=dtr)
    t0 = time.time()
    booster = lgb.train(p, dtr, num_boost_round=params["n_estimators"], valid_sets=[dva],
                        callbacks=[lgb.early_stopping(params["early_stopping_rounds"], verbose=False)])
    return booster, {"best_iteration": int(booster.best_iteration), "train_seconds": round(time.time() - t0, 2)}


class Isotonic:
    """Monotone map fit on held-out predictions. Stored as knots; applied with np.interp,
    which is exactly sklearn's IsotonicRegression(out_of_bounds='clip') prediction."""

    def __init__(self, x=None, y=None):
        self.x, self.y = x, y

    def fit(self, p, y, sample_weight=None):
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(np.asarray(p, float), np.asarray(y, float), sample_weight=sample_weight)
        self.x, self.y = iso.X_thresholds_.astype(float), iso.y_thresholds_.astype(float)
        return self

    def __call__(self, p):
        return np.interp(np.asarray(p, float), self.x, self.y)

    def to_dict(self):
        return {"x": self.x.tolist(), "y": self.y.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(np.asarray(d["x"], float), np.asarray(d["y"], float))


class BustModel:
    def __init__(self, booster, calibrator, builder, clip=(0.01, 0.99), recalibrator=None, metadata=None):
        self.booster, self.calibrator, self.builder = booster, calibrator, builder
        self.clip, self.recalibrator, self.metadata = tuple(clip), recalibrator, metadata or {}

    def predict_raw(self, X):
        return self.booster.predict(X[FEATURES], num_iteration=self.booster.best_iteration or None)

    def calibrate(self, raw):
        p = self.calibrator(raw)
        if self.recalibrator is not None:
            p = self.recalibrator(p)
        return np.clip(p, *self.clip)

    def predict(self, X):
        """Calibrated bust probability."""
        return self.calibrate(self.predict_raw(X))

    def contributions(self, X):
        """Exact TreeSHAP values (log-odds of the raw model), shape (n, n_features);
        the last column LightGBM returns (the expected value) is dropped."""
        c = self.booster.predict(X[FEATURES], num_iteration=self.booster.best_iteration or None, pred_contrib=True)
        return c[:, :-1]

    # ------------------------------------------------------------ persistence
    def save(self, d):
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(d / "booster.txt"), num_iteration=self.booster.best_iteration or None)
        (d / "calibrator.json").write_text(json.dumps(self.calibrator.to_dict()))
        (d / "features.json").write_text(json.dumps(self.builder.state_dict()))
        meta = dict(self.metadata, clip=list(self.clip))
        (d / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
        rc = d / "recalibrator.json"
        if self.recalibrator is not None:
            rc.write_text(json.dumps(self.recalibrator.to_dict()))
        elif rc.exists():
            rc.unlink()

    @classmethod
    def load(cls, d):
        d = Path(d)
        booster = lgb.Booster(model_file=str(d / "booster.txt"))
        cal = Isotonic.from_dict(json.loads((d / "calibrator.json").read_text()))
        builder = FeatureBuilder.from_state(json.loads((d / "features.json").read_text()))
        meta = json.loads((d / "metadata.json").read_text())
        rc = d / "recalibrator.json"
        recal = Isotonic.from_dict(json.loads(rc.read_text())) if rc.exists() else None
        return cls(booster, cal, builder, clip=meta.get("clip", (0.01, 0.99)), recalibrator=recal, metadata=meta)
