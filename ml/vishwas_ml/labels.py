"""The locked bust definition.

    bust = |forecast - observed| >= max(min_error_mm, P95 of |error| for this subdivision
                                         and lead day, fit on training years only)
           OR  rain/no-rain category call was wrong

Both conditions are checked independently; `trigger_reason` records which fired.
The rain/no-rain line is IMD's rainy-day threshold, 2.5 mm (config: bust.rain_threshold_mm).
"""
import numpy as np
import pandas as pd

GROUP = ["subdivision_code", "lead_day"]


def fit_thresholds(train, percentile=95, min_error_mm=25.0):
    """P-th percentile of |error| per (subdivision, lead day), floored at min_error_mm.
    Pass training-year rows only."""
    abs_err = (train["forecast_rain"] - train["observed_rain"]).abs()
    q = abs_err.groupby([train[c] for c in GROUP]).quantile(percentile / 100.0).rename("p_err").reset_index()
    q["tau"] = np.maximum(q["p_err"], min_error_mm)
    return q


def attach_tau(df, thresholds, min_error_mm=25.0):
    """Threshold for every row. A (subdivision, lead) pair with no training rows falls back
    to the median training percentile for that lead day (never its own data)."""
    out = df[GROUP].merge(thresholds, on=GROUP, how="left")
    if out["tau"].isna().any():
        fb = thresholds.groupby("lead_day")["p_err"].median()
        out["tau"] = out["tau"].fillna(np.maximum(out["lead_day"].map(fb), min_error_mm)).fillna(min_error_mm)
    return out["tau"].to_numpy(float)


def bust_labels(forecast, observed, tau, rain_threshold_mm=2.5):
    """Return (is_bust bool array, trigger_reason object array with None for non-busts)."""
    f = np.asarray(forecast, float)
    o = np.asarray(observed, float)
    magnitude = np.abs(f - o) >= np.asarray(tau, float)
    category = (f >= rain_threshold_mm) != (o >= rain_threshold_mm)
    reason = np.select([magnitude & category, magnitude, category], ["both", "magnitude", "category"], "")
    reason = np.where(reason == "", None, reason).astype(object)
    return magnitude | category, reason


def label_frame(df, thresholds, bust_cfg):
    """Labels for the rows of df (observed_rain must be present)."""
    tau = attach_tau(df, thresholds, bust_cfg["min_error_mm"])
    y, reason = bust_labels(df["forecast_rain"], df["observed_rain"], tau, bust_cfg["rain_threshold_mm"])
    return pd.DataFrame({"is_bust": y, "trigger_reason": reason, "tau": tau}, index=df.index)
