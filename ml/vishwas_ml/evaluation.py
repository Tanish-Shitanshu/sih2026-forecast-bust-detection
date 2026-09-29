"""Metrics for a rare-event classifier.

PR-AUC (average precision) is the headline number: with a 10-20% base rate, accuracy
rewards a model that never flags anything. Every table also reports the base rate, the
PR-AUC of a no-skill model.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def brier(y, p):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def ece(y, p, n_bins=10):
    """Expected calibration error with equal-count bins."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = [b for b in np.array_split(np.argsort(p, kind="stable"), n_bins) if len(b)]
    return float(sum(len(b) * abs(y[b].mean() - p[b].mean()) for b in bins) / len(p))


def reliability(y, p, n_bins=10):
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = [b for b in np.array_split(np.argsort(p, kind="stable"), n_bins) if len(b)]
    return pd.DataFrame({"mean_predicted": [p[b].mean() for b in bins],
                         "observed_rate": [y[b].mean() for b in bins], "n": [len(b) for b in bins]})


def at_threshold(y, p, t):
    y, flag = np.asarray(y, bool), np.asarray(p) >= t
    tp = int((flag & y).sum())
    prec = tp / flag.sum() if flag.any() else float("nan")
    rec = tp / y.sum() if y.any() else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if flag.any() and y.any() and (prec + rec) > 0 else float("nan")
    return {"precision": prec, "recall": rec, "f1": f1, "flagged_frac": float(flag.mean())}


def catch_rate(y, p, budget=0.10):
    """Share of busts caught if forecasters review only the top `budget` of cases."""
    y = np.asarray(y, float)
    k = max(1, int(round(budget * len(p))))
    top = np.argsort(-np.asarray(p), kind="stable")[:k]
    return float(y[top].sum() / max(1.0, y.sum()))


def core(y, p, p_ref=None, alert=0.38, watch=0.18):
    y = np.asarray(y).astype(int)
    out = {"n": int(len(y)), "busts": int(y.sum()), "base_rate": float(y.mean()) if len(y) else float("nan")}
    if len(y) == 0 or y.sum() in (0, len(y)):
        return out
    out.update(pr_auc=float(average_precision_score(y, p)), roc_auc=float(roc_auc_score(y, p)),
               brier=brier(y, p), ece=ece(y, p), catch_at_10pct=catch_rate(y, p),
               accuracy_at_0_5=float(((np.asarray(p) >= 0.5) == y).mean()))
    if p_ref is not None:
        out["brier_skill_vs_climatology"] = 1 - out["brier"] / brier(y, p_ref)
    for name, t in (("alert", alert), ("watch", watch)):
        for k, v in at_threshold(y, p, t).items():
            out[f"{k}_at_{name}"] = v
    return out


def grouped(df, by, p_col="p", y_col="y", ref_col="p_clim", **kw):
    rows = []
    for key, g in df.groupby(by, sort=True):
        r = core(g[y_col], g[p_col], g[ref_col] if ref_col in g else None, **kw)
        r[by] = key
        rows.append(r)
    cols = [by, "n", "busts", "base_rate", "pr_auc", "roc_auc", "precision_at_alert", "recall_at_alert",
            "precision_at_watch", "recall_at_watch", "brier", "brier_skill_vs_climatology", "ece", "catch_at_10pct"]
    out = pd.DataFrame(rows)
    return out[[c for c in cols if c in out.columns]]
