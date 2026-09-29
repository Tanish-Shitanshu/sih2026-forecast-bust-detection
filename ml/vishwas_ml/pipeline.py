"""End-to-end training and evaluation.

    from vishwas_ml.pipeline import train_pipeline
    summary = train_pipeline()          # uses ml/config.json

Steps: load + validate -> year split -> bust thresholds on training years -> labels ->
features (OOF bust rate on training rows) -> LightGBM (early stopping on the calibration
year) -> isotonic calibration on the calibration year -> test-year evaluation with
baselines -> SHAP importance -> analog library -> save bundle.
"""
import hashlib
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .analogs import AnalogIndex, importance_weights
from .config import ML_DIR, load_config, load_meta, subdivision_codes
from .evaluation import core, grouped, reliability
from .events import tag_events
from .explain import global_importance
from .features import FEATURES, FeatureBuilder
from .labels import fit_thresholds, label_frame
from .model import BustModel, Isotonic, train_booster
from .schema import load_pairs, validate
from .splits import year_split
from .trust import feature_ranges


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _rel(path):
    try:
        return Path(path).resolve().relative_to(ML_DIR).as_posix()
    except ValueError:
        return Path(path).name


def prepare(cfg, data_path=None, log=print):
    path = Path(data_path or cfg["data_path"])
    meta = load_meta(cfg)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = load_pairs(path)
        report = validate(df, subdivision_codes(meta))
    report["warnings"] = [str(w.message) for w in caught]
    for w in report["warnings"]:
        log(f"  warning: {w}")
    timing = report["era5_timing_detected"] if cfg["era5_timing"] == "auto" else cfg["era5_timing"]
    if timing == "valid":
        log("  ERA5 columns vary with lead_day: treating them as valid-date values and reading the "
            "issue-date state from earlier rows (valid-date reanalysis would leak the outcome)")
    df = df[df["observed_rain"].notna()].reset_index(drop=True)
    split = year_split(df, cfg["split"])
    return df, report, timing, split, meta, path


def evaluate(model, df, split, cfg, out_dir=None, log=print):
    """Test-year metrics for the model and two baselines. Writes CSV/JSON/PNG to
    out_dir/evaluation when out_dir is given."""
    b = model.builder
    lab = label_frame(df, b.thresholds, cfg["bust"])
    y = lab["is_bust"].to_numpy()
    X = b.transform(df, oof=split["train"])
    p = model.predict(X)
    tr, ca, te = split["train"], split["calib"], split["test"]

    # Baseline 1: climatology = historical bust rate for (subdivision, lead, month).
    p_clim = X["hist_bust_rate"].to_numpy()
    # Baseline 2: forecast amount + lead day only, same booster + isotonic recipe.
    base_cols = ["forecast_rain", "lead_day"]
    bb, _ = train_booster(X.loc[tr, base_cols], y[tr], X.loc[ca, base_cols], y[ca],
                          dict(cfg["lightgbm"], n_estimators=500))
    iso = Isotonic().fit(bb.predict(X.loc[ca, base_cols]), y[ca])
    p_fc = iso(bb.predict(X[base_cols]))

    t = pd.DataFrame({"y": y[te], "p": p[te], "p_clim": p_clim[te], "p_fc": p_fc[te],
                      "lead_day": df.loc[te, "lead_day"].to_numpy(),
                      "subdivision_code": df.loc[te, "subdivision_code"].to_numpy(),
                      "trigger_reason": lab.loc[te, "trigger_reason"].to_numpy(),
                      "month": df.loc[te, "date"].dt.month.to_numpy()})
    tiers = {tt["key"]: tt["max"] for tt in load_meta(cfg)["tiers"]}
    kw = dict(alert=tiers["yellow"], watch=tiers["green"])  # Orange+ = alert, Yellow+ = watch
    overall = {"model": core(t["y"], t["p"], t["p_clim"], **kw),
               "baseline_climatology": core(t["y"], t["p_clim"], t["p_clim"], **kw),
               "baseline_forecast_amount_only": core(t["y"], t["p_fc"], t["p_clim"], **kw)}
    per_lead = grouped(t, "lead_day", **kw)
    per_lead["pr_auc_climatology"] = [core(g["y"], g["p_clim"])["pr_auc"] for _, g in t.groupby("lead_day")]
    per_lead["pr_auc_forecast_only"] = [core(g["y"], g["p_fc"])["pr_auc"] for _, g in t.groupby("lead_day")]
    per_sub = grouped(t, "subdivision_code", **kw)
    busts = t[t["y"] == 1]
    recall_by_trigger = {r: float((g["p"] >= kw["alert"]).mean()) for r, g in busts.groupby("trigger_reason")}
    rel = reliability(t["y"], t["p"])
    rel_raw = reliability(t["y"], model.predict_raw(X[te]))
    metrics = {"split_years": split["years"], "n_test": int(te.sum()), "overall": overall,
               "recall_at_alert_by_trigger": recall_by_trigger,
               "per_lead": per_lead.round(4).to_dict(orient="records"),
               "reliability_calibrated": rel.round(4).to_dict(orient="records"),
               "reliability_raw": rel_raw.round(4).to_dict(orient="records")}
    if out_dir:
        ev = Path(out_dir) / "evaluation"
        ev.mkdir(parents=True, exist_ok=True)
        per_lead.round(4).to_csv(ev / "per_lead.csv", index=False)
        per_sub.round(4).to_csv(ev / "per_subdivision.csv", index=False)
        (ev / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
        _plots(ev, t, rel, rel_raw, per_lead)
    return metrics, per_sub, (X, y, p, lab)


def _plots(ev, t, rel, rel_raw, per_lead):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot([0, 1], [0, 1], color="#999", lw=1, ls="--")
    ax[0].plot(rel_raw["mean_predicted"], rel_raw["observed_rate"], "o-", color="#bbb", label="raw LightGBM")
    ax[0].plot(rel["mean_predicted"], rel["observed_rate"], "o-", color="#1A4E9B", label="isotonic-calibrated")
    ax[0].set(xlabel="predicted bust probability", ylabel="observed bust frequency",
              title="Reliability, test year", xlim=(0, 1), ylim=(0, 1))
    ax[0].legend(frameon=False)
    x = per_lead["lead_day"].to_numpy()
    ax[1].plot(x, per_lead["pr_auc"], "o-", color="#1A4E9B", label="model")
    ax[1].plot(x, per_lead["pr_auc_forecast_only"], "s-", color="#E8720C", label="forecast amount only")
    ax[1].plot(x, per_lead["pr_auc_climatology"], "^-", color="#888", label="climatology")
    ax[1].plot(x, per_lead["base_rate"], ":", color="#C62828", label="no-skill (base rate)")
    ax[1].set(xlabel="lead day", ylabel="PR-AUC", title="PR-AUC by lead day, test year", ylim=(0, 1))
    ax[1].legend(frameon=False, fontsize=8)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ev / "evaluation.png", dpi=130)
    plt.close(fig)


def train_pipeline(cfg=None, data_path=None, out_dir=None, log=print):
    t0 = time.time()
    cfg = cfg or load_config()
    out_dir = Path(out_dir or cfg["models_dir"])
    log(f"loading {data_path or cfg['data_path']}")
    df, report, timing, split, meta, path = prepare(cfg, data_path, log)
    tr, ca, te = split["train"], split["calib"], split["test"]
    log(f"  {len(df):,} labelled rows; train {split['years']['train']}, calibration {split['years']['calib']}, "
        f"test {split['years']['test']}")

    thr = fit_thresholds(df[tr], cfg["bust"]["percentile"], cfg["bust"]["min_error_mm"])
    lab = label_frame(df, thr, cfg["bust"])
    y = lab["is_bust"].to_numpy()
    agree = float((y == df["is_bust"].to_numpy()).mean())
    log(f"  bust rate {y.mean():.3f} (train {y[tr].mean():.3f}, test {y[te].mean():.3f}); "
        f"agreement with provided is_bust: {agree:.4f}")

    builder = FeatureBuilder(meta, cfg["bust"], timing).fit(df[tr], y[tr], thr)
    X = builder.transform(df, oof=tr)
    booster, info = train_booster(X[tr], y[tr], X[ca], y[ca], cfg["lightgbm"])
    log(f"  LightGBM: {info['best_iteration']} trees in {info['train_seconds']} s")
    raw = booster.predict(X, num_iteration=booster.best_iteration)
    cal = Isotonic().fit(raw[ca], y[ca])
    model = BustModel(booster, cal, builder, clip=cfg["probability_clip"])

    metrics, per_sub, _ = evaluate(model, df, split, cfg, out_dir, log)
    m = metrics["overall"]
    log(f"  test PR-AUC {m['model']['pr_auc']:.3f} (climatology {m['baseline_climatology']['pr_auc']:.3f}, "
        f"forecast-only {m['baseline_forecast_amount_only']['pr_auc']:.3f}, base rate {m['model']['base_rate']:.3f})")

    rng = np.random.default_rng(0)
    samp = rng.choice(np.where(te)[0], min(20000, int(te.sum())), replace=False)
    imp = global_importance(model.contributions(X.iloc[samp]))

    vmonth = (df["date"] + pd.to_timedelta(df["lead_day"], "D")).dt.month
    lib = df[["date", "subdivision_code", "lead_day", "forecast_rain", "observed_rain"]].copy()
    lib["region"] = df["subdivision_code"].map(builder.region_of).to_numpy()
    lib["is_bust"], lib["trigger_reason"] = y, lab["trigger_reason"].to_numpy()
    lib["event"] = tag_events(X, df["subdivision_code"], vmonth, meta["event_sets"]).to_numpy()
    analogs = AnalogIndex().fit(X, lib, importance_weights(imp["features"]))

    model.metadata = {
        "version": __version__, "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_path": _rel(path), "data_sha256_16": _sha(path), "data_report": report, "era5_timing": timing,
        "split_years": split["years"], "bust_definition": cfg["bust"], "label_agreement_with_provided": agree,
        "features": FEATURES, "lightgbm": info, "importance": imp,
        "trust_ranges": feature_ranges(X[tr]), "summary_metrics": {k: v for k, v in m["model"].items()},
        "baselines": {k: m[k] for k in ("baseline_climatology", "baseline_forecast_amount_only")},
    }
    model.save(out_dir)
    analogs.save(out_dir)
    pd.DataFrame({"p": model.predict(X[ca]), "y": y[ca]}).to_parquet(out_dir / "calibration_set.parquet", index=False)
    log(f"saved bundle to {out_dir} in {time.time() - t0:.1f} s")
    return {"metrics": metrics, "per_subdivision": per_sub, "importance": imp, "out_dir": str(out_dir)}
