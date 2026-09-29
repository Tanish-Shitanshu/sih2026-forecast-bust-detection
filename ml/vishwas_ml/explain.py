"""Turn SHAP values into plain-language factors for the frontend.

Features are grouped (all pressure features form one "pressure" factor, and so on); a
group's contribution is the sum of its features' SHAP values in log-odds. The share of
total |SHAP| gives `contribution_percent`, the sign gives `direction`, and the sentence
is written from the row's actual values. Wording avoids raw ERA5 units so it holds
whatever units the data pipeline ships.

SHAP explains the raw classifier. Calibration is a monotone map applied after it, so the
direction and ranking of the factors carry over to the calibrated probability.
"""
import calendar

import numpy as np

from .features import FEATURE_GROUPS, FEATURES

GROUPS = list(dict.fromkeys(FEATURE_GROUPS.values()))
_GIDX = {g: [FEATURES.index(f) for f, gg in FEATURE_GROUPS.items() if gg == g] for g in GROUPS}


def group_contributions(contrib):
    """(n, n_features) SHAP -> (n, n_groups) summed per group, columns in GROUPS order."""
    return np.column_stack([contrib[:, _GIDX[g]].sum(1) for g in GROUPS])


def _level(z, hi="above", lo="below"):
    if z is None or not np.isfinite(z):
        return None
    if z >= 1.5:
        return f"well {hi} normal"
    if z >= 0.5:
        return f"{hi} normal"
    if z <= -1.5:
        return f"well {lo} normal"
    if z <= -0.5:
        return f"{lo} normal"
    return "near normal"


def _ratio(r):
    if not np.isfinite(r):
        return "compared with"
    if r > 0.7:
        return "well above"
    if r > 0.25:
        return "above"
    if r < -0.7:
        return "well below"
    if r < -0.25:
        return "below"
    return "close to"


def describe(group, x, ctx):
    """One sentence for a factor group. x: dict of the row's features; ctx: names/month."""
    month = calendar.month_name[ctx["month"]]
    L = int(x["lead_day"])
    fc = x["forecast_rain"]
    if group == "lead":
        return f"Day {L} lead time" + (", where forecast errors grow" if L >= 6 else "")
    if group == "amount":
        thr = ctx["rain_thr"]
        if abs(fc - thr) < 1.0:
            return f"Forecast of {fc:.1f} mm sits right at the {thr:g} mm rain/no-rain line, so a small error flips the call"
        if fc < 0.5:
            return f"Little or no rain forecast ({fc:.1f} mm), {_ratio(x['fc_clim_ratio'])} the usual for {month}"
        if fc < thr:
            return f"Light rain forecast ({fc:.1f} mm), below the {thr:g} mm rain/no-rain line"
        return f"Forecast of {fc:.0f} mm, {_ratio(x['fc_clim_ratio'])} the usual for {month}"
    if group == "run_change":
        j, gap = x["fc_jump"], x["fc_jump_gap"]
        if not np.isfinite(j):
            return "No earlier run covers this day to compare against"
        if abs(j) < 1:
            return f"Forecast barely changed since the run {int(gap)} day{'s' if gap != 1 else ''} earlier"
        return f"Forecast changed by {j:+.0f} mm since the run {int(gap)} day{'s' if gap != 1 else ''} earlier"
    if group == "lead_consistency":
        s = x["fc_neighbor_std"]
        word = "sharply" if np.isfinite(s) and s > 10 else "moderately" if np.isfinite(s) and s > 3 else "little"
        return f"Forecast rain varies {word} across neighbouring days of this run"
    if group == "ensemble":
        s = x["fc_spread"]
        if not np.isfinite(s):
            return "No ensemble spread available for this forecast"
        word = "disagree strongly" if x["fc_spread_rel"] > 1 else "differ somewhat" if x["fc_spread_rel"] > 0.4             else "agree closely"
        return f"Ensemble members {word} (spread {s:.0f} mm)"
    if group == "regional":
        return f"Forecast is {x['fc_region_dev']:+.0f} mm from the {ctx['region_name']} average"
    if group == "pressure":
        lv = _level(x["mslp_z"])
        if lv is None:
            return "Pressure data unavailable at issue time"
        low = min(x["mslp_z"], x["mslp_region_z"] if np.isfinite(x["mslp_region_z"]) else 0) < -1
        return f"Sea-level pressure {lv}" + (", suggesting a low-pressure system" if low else "")
    if group == "moisture":
        lv = _level(x["dewpoint_z"])
        if lv is None:
            return "Moisture data unavailable at issue time"
        return f"Surface moisture {lv} (dewpoint depression {x['dewpoint_depression']:.1f} degC)"
    if group == "temperature":
        lv = _level(x["temp_z"])
        return f"Temperature {lv}" if lv else "Temperature data unavailable at issue time"
    if group == "wind":
        z = x["wind_speed_z"]
        if not np.isfinite(z):
            return "Wind data unavailable at issue time"
        word = ("much stronger than" if z >= 1.5 else "stronger than" if z >= 0.5 else
                "much weaker than" if z <= -1.5 else "weaker than" if z <= -0.5 else "near")
        return f"Surface winds {word} normal"
    if group == "recent_rain":
        r = x["era5_tp_rel"]
        if not np.isfinite(r):
            return "Rainfall on the issue day unknown"
        word = "well above" if r > np.log1p(3) else "above" if r > np.log1p(1.5) else \
            "below" if r < np.log1p(0.5) else "close to"
        return f"Rain on the issue day {word} the usual for {month}"
    if group == "history":
        return (f"Forecasts here bust {x['hist_bust_rate']:.0%} of the time at Day {L} in {month}")
    if group == "season":
        return f"Seasonal pattern for {month} in {ctx['region_name']}"
    return group


def explain_rows(X, contrib, ctxs, top=3):
    """Top factors per row as [{feature, contribution_percent, direction}] (frontend shape)."""
    G = group_contributions(contrib)
    tot = np.abs(G).sum(1, keepdims=True)
    tot[tot == 0] = 1
    share = np.abs(G) / tot
    recs = X.to_dict(orient="records")
    out = []
    for i in range(len(G)):
        order = np.argsort(-np.abs(G[i]), kind="stable")[:top]
        out.append([{"feature": describe(GROUPS[j], recs[i], ctxs[i]),
                     "contribution_percent": int(round(share[i, j] * 100)),
                     "direction": "raises" if G[i, j] > 0 else "lowers"} for j in order])
    return out


def global_importance(contrib):
    """Mean |SHAP| per feature and per group, for the model card and analog weights."""
    f = dict(zip(FEATURES, np.abs(contrib).mean(0).round(5).tolist()))
    g = dict(zip(GROUPS, np.abs(group_contributions(contrib)).mean(0).round(5).tolist()))
    return {"features": dict(sorted(f.items(), key=lambda kv: -kv[1])),
            "groups": dict(sorted(g.items(), key=lambda kv: -kv[1]))}
