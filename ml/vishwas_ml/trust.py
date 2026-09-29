"""Model self-confidence: how much precedent the model has for this case.

Separate from bust probability. Four reliability flags, each scored 0..1:
  sparse_history   few training busts for this subdivision and lead day
  unfamiliar       nearest historical analogs are unusually far away (novel pattern)
  out_of_range     inputs outside the range seen in training
  analog_conflict  the model's probability disagrees with what the analogs did
Uncertainty u = 1 - (1 - base) * prod(1 - weight_i * flag_i); self-confidence = 1 - u.
Level labels use the frontend's VIO thresholds on u.
"""
import numpy as np

WEIGHTS = {"sparse_history": 0.6, "unfamiliar": 0.6, "out_of_range": 0.5, "analog_conflict": 0.45}
BASE = 0.10
REASONS = {
    "sparse_history": "Sparse historical analog data",
    "unfamiliar": "Rare synoptic configuration for this lead time",
    "out_of_range": "Pattern outside training climatology",
    "analog_conflict": "Model and historical analogs disagree",
    None: "Pattern well represented in training history",
}
RANGE_FEATURES = ["forecast_rain", "mslp_z", "dewpoint_z", "temp_z", "wind_speed_z", "dewpoint_depression_z"]


def feature_ranges(X):
    return {c: [float(np.nanpercentile(X[c], 0.5)), float(np.nanpercentile(X[c], 99.5))]
            for c in RANGE_FEATURES if X[c].notna().any()}


def flags(support_busts, novelty, X, ranges, p, analog_rate, min_busts=30):
    n = len(p)
    f = {}
    f["sparse_history"] = np.clip((min_busts - np.asarray(support_busts, float)) / min_busts, 0, 1)
    nov = np.nan_to_num(np.asarray(novelty, float), nan=1.0)  # no analogs at all = unfamiliar
    f["unfamiliar"] = np.clip((nov - 0.90) / 0.10, 0, 1)
    out = np.zeros(n)
    for c, (lo, hi) in ranges.items():
        v = X[c].to_numpy(float)
        out += ((v < lo) | (v > hi)) & np.isfinite(v)
    f["out_of_range"] = np.clip(out / max(len(ranges), 1) * 3, 0, 1)
    gap = np.abs(np.asarray(p, float) - np.nan_to_num(np.asarray(analog_rate, float), nan=np.asarray(p, float)))
    f["analog_conflict"] = np.clip((gap - 0.15) / 0.35, 0, 1)
    return f


def combine(f):
    keep = np.ones_like(next(iter(f.values())))
    for k, v in f.items():
        keep = keep * (1 - WEIGHTS[k] * v)
    u = 1 - (1 - BASE) * keep
    weighted = np.column_stack([WEIGHTS[k] * f[k] for k in f])
    names = list(f)
    top = weighted.argmax(1)
    reason_key = [names[t] if weighted[i, t] > 0.15 else None for i, t in enumerate(top)]
    active = [[names[j] for j in range(len(names)) if weighted[i, j] > 0.15] for i in range(len(u))]
    return u, reason_key, active
