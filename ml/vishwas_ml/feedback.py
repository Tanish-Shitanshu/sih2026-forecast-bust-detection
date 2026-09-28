"""Forecaster outcomes -> periodic recalibration.

The frontend records an outcome per flagged prediction: correct, incorrect or partial.
Interpretation used here (from the frontend's sample log, where high predicted bust
probabilities pair with "incorrect"): the outcome describes the issued forecast.
    incorrect -> the forecast busted (label 1)
    correct   -> it held (label 0)
    partial   -> excluded by default (config feedback.partial_as can map it to 0 or 1)

Only outcomes with status "approved" are used, as on the frontend. Recalibration fits a
second isotonic map on top of the current calibrated probability, on the calibration-
year predictions plus the approved outcomes (weight feedback.weight each), so a handful of
entries cannot swing the model. Below feedback.min_outcomes nothing changes.
When real observations arrive, a full retrain (scripts/train.py) is the stronger update.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .model import Isotonic

OUTCOMES = {"correct": 0, "incorrect": 1, "partial": None}


def validate_outcome(payload):
    """Check a POST /api/v1/outcomes body. Raises ValueError with a 422-style message."""
    if payload.get("outcome") not in OUTCOMES:
        raise ValueError("outcome must be correct, incorrect or partial")
    L = payload.get("lead_day")
    if not isinstance(L, int) or not 1 <= L <= 10:
        raise ValueError("lead_day must be an integer from 1 to 10")
    note = payload.get("note")
    if note is not None and len(str(note)) > 80:
        raise ValueError("note must be at most 80 characters")
    return True


def append_outcome(path, rec):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_outcomes(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["id", "subdivision", "lead_day", "predicted_bust_probability", "outcome", "status"])
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_json(path, lines=True)


def outcome_labels(df, partial_as=None):
    """Approved outcomes as (predicted probability, label) with unusable rows dropped."""
    d = df[df["status"].astype(str).str.lower() == "approved"].copy()
    m = dict(OUTCOMES, partial=partial_as)
    d["label"] = d["outcome"].map(m)
    d = d[d["label"].notna() & d["predicted_bust_probability"].notna()]
    return d["predicted_bust_probability"].to_numpy(float), d["label"].to_numpy(float)


def recalibrate(model, outcomes, calib_set, fb_cfg):
    """Return (recalibrator or None, summary). model: BustModel; outcomes: DataFrame with
    predicted_bust_probability, outcome, status; calib_set: DataFrame p, y from training."""
    p_fb, y_fb = outcome_labels(outcomes, fb_cfg.get("partial_as"))
    summary = {"approved_used": int(len(p_fb)), "min_outcomes": fb_cfg["min_outcomes"]}
    if len(p_fb) < fb_cfg["min_outcomes"]:
        summary["status"] = "skipped: not enough approved outcomes"
        return None, summary
    p = np.concatenate([calib_set["p"].to_numpy(float), p_fb])
    y = np.concatenate([calib_set["y"].to_numpy(float), y_fb])
    w = np.concatenate([np.ones(len(calib_set)), np.full(len(p_fb), float(fb_cfg["weight"]))])
    rc = Isotonic().fit(p, y, sample_weight=w)
    summary["status"] = "fitted"
    summary["feedback_bust_rate"] = float(y_fb.mean())
    summary["feedback_mean_predicted"] = float(p_fb.mean())
    return rc, summary
