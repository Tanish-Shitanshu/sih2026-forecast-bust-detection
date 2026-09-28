"""Weather-event tag for a flagged subdivision: a transparent rules engine, not ML.

Uses the frontend's event names and its per-subdivision candidate sets (EV_* in
frontend/index.html), then picks the candidate best supported by the issue-date ERA5
anomalies, the forecast amount and the season of the valid date. Heuristic by design;
the label only says which kind of situation the bust risk sits in.
"""
import numpy as np
import pandas as pd

HEAVY_MM = 64.5  # IMD "heavy rain" threshold for 24 h


def candidates(code, sets):
    """Same order and logic as eventCands() in frontend/index.html."""
    c = []
    if code in sets["NW"]:
        c.append("western disturbance")
    if code in sets["HEAT"]:
        c.append("heat wave")
    if code in sets["COAST"]:
        c.append("cyclone")
    if code not in sets["NODEP"]:
        c.append("monsoon depression")
    if code in sets["HEAVY"]:
        c.append("heavy rainfall event")
    c.append("break or active monsoon phase")
    return c


def tag_events(X, codes, valid_month, sets):
    """Event name per row (for every row; callers blank it for Green)."""
    X = X.reset_index(drop=True)
    codes = np.asarray(codes)
    m = np.asarray(valid_month)
    g = lambda c: X[c].fillna(0).to_numpy(float)  # noqa: E731
    mslp, mreg, dew, temp, dpd, wind, tp = (g(c) for c in ("mslp_z", "mslp_region_z", "dewpoint_z", "temp_z",
                                                          "dewpoint_depression_z", "wind_speed_z", "era5_tp_rel"))
    fc = X["forecast_rain"].to_numpy(float)
    jjas = np.isin(m, [6, 7, 8, 9])
    ninf = -np.inf
    score = {
        "cyclone": np.where(np.isin(m, [4, 5, 6, 10, 11, 12]) & (mslp < -1.8) & (wind > 0.8),
                            3 - mslp + 0.7 * wind, ninf),
        "monsoon depression": np.where(jjas & (mreg < -0.8), 2 - mreg + 0.3 * dew, ninf),
        "western disturbance": np.where(np.isin(m, [11, 12, 1, 2, 3, 4]) & ((mslp < -0.5) | (tp > 0.5)),
                                        1.5 - mslp, ninf),
        "heat wave": np.where(np.isin(m, [3, 4, 5, 6]) & (temp > 1.2) & (dpd > 0.3), 1.5 + temp, ninf),
        "heavy rainfall event": np.where(fc >= 35.5, 1.0 + np.log1p(fc) / np.log1p(HEAVY_MM), ninf),
        "break or active monsoon phase": np.where(jjas, 0.8, ninf),
    }
    out = []
    for i, code in enumerate(codes):
        cand = candidates(code, sets)
        best = max(cand, key=lambda e: score[e][i])
        if not np.isfinite(score[best][i]):  # nothing qualifies: season fallback, else no tag
            if "western disturbance" in cand and m[i] in (11, 12, 1, 2, 3):
                best = "western disturbance"
            else:
                best = None
        out.append(best)
    return pd.Series(out, dtype=object)
