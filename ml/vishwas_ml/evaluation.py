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


# ---------------------------------------------------------------- uncertainty

def fast_ap(y, p):
    """Average precision (same tie handling as sklearn), fast enough for bootstrap loops."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    order = np.argsort(-p, kind="mergesort")
    ys, ps = y[order], p[order]
    tp = np.cumsum(ys)
    last = np.r_[np.where(np.diff(ps))[0], len(ps) - 1]  # last index of each tied score
    tp = tp[last]
    if tp[-1] == 0:
        return float("nan")
    prec = tp / (last + 1)
    rec = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0, rec]) * prec))


def block_bootstrap(y, preds, blocks, reps=500, seed=0, level=0.95):
    """CIs for PR-AUC by resampling whole issue dates (rows from one forecast run are correlated,
    so resampling rows would make intervals too narrow).

    preds: {name: probabilities}. Returns {name: {"pr_auc", "lo", "hi"}} and, for every pair
    (a, b), the CI of pr_auc[a] - pr_auc[b] with the share of resamples where a beats b."""
    y = np.asarray(y, float)
    uniq, inv = np.unique(np.asarray(blocks), return_inverse=True)
    order = np.argsort(inv, kind="stable")
    groups = np.split(order, np.cumsum(np.bincount(inv))[:-1])
    rng = np.random.default_rng(seed)
    samples = {k: [] for k in preds}
    for _ in range(reps):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if y[idx].sum() == 0:
            continue
        for k, p in preds.items():
            samples[k].append(fast_ap(y[idx], np.asarray(p)[idx]))
    a = (1 - level) / 2 * 100
    out = {k: {"pr_auc": fast_ap(y, preds[k]), "lo": float(np.nanpercentile(v, a)),
               "hi": float(np.nanpercentile(v, 100 - a))} for k, v in samples.items()}
    names = list(preds)
    diffs = {}
    for i, ka in enumerate(names):
        for kb in names[i + 1:]:
            d = np.asarray(samples[ka]) - np.asarray(samples[kb])
            diffs[f"{ka} - {kb}"] = {"mean": float(np.nanmean(d)), "lo": float(np.nanpercentile(d, a)),
                                     "hi": float(np.nanpercentile(d, 100 - a)), "p_better": float(np.mean(d > 0))}
    return {"n_blocks": int(len(uniq)), "reps": reps, "pr_auc": out, "differences": diffs}
