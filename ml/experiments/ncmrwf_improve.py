"""Model-improvement experiments for the NCMRWF track (weak spots from the first evaluation).

Protocol (no tuning on the test years):
  * SELECTION: rows from 2013-2015 are dropped before anything runs. Candidates train on
    1993-2008, calibrate on 2009 and are compared on VALIDATION years 2010-2012.
  * FINAL: the variant chosen on validation for each item is retrained on the standard split
    (train 1993-2011, calibrate 2012) and evaluated ONCE on the test years 2013-2015, next to
    the baseline trained the same way. Block-bootstrap CIs over issue dates.
Bust thresholds are always fit on the run's own training years and never shared with GEFS.
Every added feature is known at 00Z on the issue date.

    python ml/experiments/ncmrwf_improve.py select   # validation table for every variant
    python ml/experiments/ncmrwf_improve.py final --variants base,<name>,...
"""
import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML))
from vishwas_ml.config import load_config, load_meta  # noqa: E402
from vishwas_ml.evaluation import block_bootstrap, fast_ap  # noqa: E402
from vishwas_ml.features import FEATURES, FeatureBuilder  # noqa: E402
from vishwas_ml.labels import fit_thresholds, label_frame  # noqa: E402
from vishwas_ml.model import Isotonic  # noqa: E402
from vishwas_ml.schema import load_pairs  # noqa: E402

SIH = ML.parents[1]
FC_CACHE = SIH / "s2s_raw" / "ncmrwf_fc_1993_2015.parquet"      # per-lead S2S subdivision fields
IMD_DAILY = SIH / "s2s_raw" / "imd_subdivision_daily_1992_2016.parquet"
OUT = ML / "experiments" / "results"
COASTAL_WEAK = ["CST.KA", "KL", "KNK/GA"]
ALERT = 0.38                                                   # Orange+ (frontend tier boundary)

SELECT = {"train": list(range(1993, 2009)), "calib": [2009], "eval": [2010, 2011, 2012]}
FINAL = {"train": list(range(1993, 2012)), "calib": [2012], "eval": [2013, 2014, 2015]}


# ---------------------------------------------------------------- data

def imd_daily():
    """IMD subdivision daily rain (date = IMD date), built once from the imdlib files."""
    if IMD_DAILY.exists():
        return pd.read_parquet(IMD_DAILY)
    sys.path.insert(0, str(ML / "scripts"))
    from build_s2s_pairs import imd_subdivision_daily
    from vishwas_ml.subdivision_weights import load_subdivisions
    subs = load_subdivisions(SIH / "geo_data" / "indian_met_zones.v2")
    d = imd_subdivision_daily(SIH / "geo_data" / "imd", list(range(1992, 2017)), subs)
    d = d.rename(columns={"valid_date": "date"}) if "valid_date" in d else d
    d.to_parquet(IMD_DAILY, index=False)
    return d


def load():
    cfg = load_config(source="ncmrwf")
    df = load_pairs(cfg["data_path"])
    return cfg, df


# ---------------------------------------------------------------- extra features (issue-time only)

def add_extra(df, X, groups, ctx):
    """Adds feature groups to X (same row order as df). ctx holds train-fitted tables."""
    X = X.copy()
    if "lead_state" in groups:
        # NCMRWF's own forecast of MSLP / surface pressure / winds around the valid day (model output,
        # available at issue), z-scored per subdivision x lead with train-year statistics.
        fcs = ctx["fc_state"]
        m = df[["date", "subdivision_code", "lead_day"]].merge(fcs, on=["date", "subdivision_code", "lead_day"],
                                                              how="left")
        for c in ("mslp", "surface_pressure", "wind_u10", "wind_v10"):
            st = ctx["lead_state_stats"][c]
            k = m[["subdivision_code", "lead_day"]].merge(st, on=["subdivision_code", "lead_day"], how="left")
            X[f"{c}_lead_z"] = ((m[c] - k["mu"]) / k["sd"]).to_numpy(float)
        X["mslp_lead_minus_issue_z"] = X["mslp_lead_z"] - X["mslp_z"]
        X["wind_speed_lead_z"] = np.hypot(X["wind_u10_lead_z"], X["wind_v10_lead_z"])
    if "antecedent" in groups:
        # IMD rain over the days before issue (IMD date <= issue - 1, complete before 00Z)
        obs = ctx["imd"].set_index(["subdivision_code", "date"])["observed_rain_mm"]
        keys = df[["subdivision_code", "date"]]
        for n in (1, 3, 7):
            tot = np.zeros(len(df)); wet = np.zeros(len(df)); cnt = np.zeros(len(df))
            for k in range(1, n + 1):
                v = obs.reindex(pd.MultiIndex.from_arrays([keys["subdivision_code"],
                                                           keys["date"] - pd.Timedelta(days=k)])).to_numpy(float)
                ok = np.isfinite(v)
                tot += np.where(ok, v, 0); wet += np.where(ok, v >= 2.5, 0); cnt += ok
            X[f"imd_prev{n}_mean"] = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
            if n > 1:
                X[f"imd_prev{n}_wetfrac"] = np.where(cnt > 0, wet / np.maximum(cnt, 1), np.nan)
        X["fc_minus_recent_obs"] = df["forecast_rain"].to_numpy(float) - X["imd_prev3_mean"]
        X["fc_wet_vs_recent_wet"] = (df["forecast_rain"].to_numpy(float) >= 2.5).astype(float) - X["imd_prev3_wetfrac"]
    if "static" in groups:
        # terrain + coast: elevation from NCMRWF's own pressure fields (hypsometric, train-year means),
        # coastal flag from the frontend's coastal event set
        X["elevation_proxy_m"] = df["subdivision_code"].map(ctx["elev"]).to_numpy(float)
        X["coastal"] = df["subdivision_code"].isin(ctx["coastal"]).to_numpy(float)
        X["coastal_x_fc"] = X["coastal"] * X["fc_log1p"]
        X["elev_x_fc"] = X["elevation_proxy_m"] / 1000 * X["fc_log1p"]
    if "interactions" in groups:
        X["fc_x_wetfrac"] = X["fc_log1p"] * X["clim_wet_frac"]
        X["near_line_x_lead"] = (np.abs(X["fc_minus_rain_threshold"]) < 2).astype(float) * X["lead_day"]
        X["fc_region_ratio"] = np.log1p(df["forecast_rain"].to_numpy(float)) - np.log1p(X["fc_region_mean"])
    return X


def fit_ctx(df, tr, meta):
    ctx = {"coastal": set(meta["event_sets"]["COAST"])}
    if FC_CACHE.exists():
        fcs = pd.read_parquet(FC_CACHE)[["date", "subdivision_code", "lead_day", "mslp", "surface_pressure",
                                          "wind_u10", "wind_v10"]]
        fcs["date"] = pd.to_datetime(fcs["date"])
        ctx["fc_state"] = fcs
        trf = fcs[fcs["date"].dt.year.isin(df.loc[tr, "date"].dt.year.unique())]
        ctx["lead_state_stats"] = {c: trf.groupby(["subdivision_code", "lead_day"])[c].agg(mu="mean", sd="std")
                                   .reset_index().assign(sd=lambda t: t["sd"].clip(lower=1e-6))
                                   for c in ("mslp", "surface_pressure", "wind_u10", "wind_v10")}
    st = df.loc[tr].groupby("subdivision_code")[["mslp", "surface_pressure"]].mean()
    ctx["elev"] = (8400 * np.log(st["mslp"] / st["surface_pressure"])).to_dict()
    ctx["imd"] = imd_daily()
    ctx["imd"]["date"] = pd.to_datetime(ctx["imd"]["date"])
    fit_ctx.cache = ctx
    return ctx


# ---------------------------------------------------------------- one run

def lgb_params(cfg, extra=None):
    p = cfg["lightgbm"]
    out = dict(objective="binary", metric="binary_logloss", verbose=-1, num_threads=0,
               num_leaves=p["num_leaves"], learning_rate=p["learning_rate"], min_child_samples=p["min_child_samples"],
               feature_fraction=p["feature_fraction"], bagging_fraction=p["bagging_fraction"],
               bagging_freq=p["bagging_freq"], lambda_l2=p["lambda_l2"], seed=p["seed"])
    out.update(extra or {})
    return out


def run(cfg, meta, df, years, v, gefs=None):
    """v: variant dict. Returns (eval frame with p / y / trigger / lead / code / date, info)."""
    yr = df["date"].dt.year
    tr, ca, ev = yr.isin(years["train"]).to_numpy(), yr.isin(years["calib"]).to_numpy(), yr.isin(years["eval"]).to_numpy()
    thr = fit_thresholds(df[tr], cfg["bust"]["percentile"], cfg["bust"]["min_error_mm"])
    lab = label_frame(df, thr, cfg["bust"])
    y = lab["is_bust"].to_numpy().astype(int)
    builder = FeatureBuilder(meta, cfg["bust"], "issue").fit(df[tr], y[tr], thr)
    X = builder.transform(df, oof=tr)
    groups = v.get("groups", [])
    if groups:
        X = add_extra(df, X, groups, fit_ctx(df, tr, meta))
    cols = [c for c in X.columns if c not in v.get("drop", [])]

    w = np.ones(len(df))
    mag = lab["trigger_reason"].isin(["magnitude", "both"]).to_numpy()
    if v.get("mag_weight"):
        w[mag] = v["mag_weight"]
    if v.get("pos_weight"):
        w[y == 1] *= v["pos_weight"]
    if v.get("coastal_weight"):
        w[df["subdivision_code"].isin(COASTAL_WEAK).to_numpy()] *= v["coastal_weight"]

    params = lgb_params(cfg, v.get("params"))
    Xtr, ytr, wtr = X.loc[tr, cols], y[tr], w[tr]
    init = None
    if gefs is not None and v.get("gefs"):
        gx = gefs["X"]
        g_groups = [grp for grp in groups if grp in ("antecedent", "interactions", "static")]
        if g_groups:
            gctx = dict(fit_ctx.cache, imd=gefs["imd"])
            gx = add_extra(gefs["df"], gx, g_groups, gctx)
        Xg, yg = gx.reindex(columns=cols), gefs["y"]  # NCMRWF-only columns (lead_state) stay missing
        if v["gefs"] == "pool":
            src = np.r_[np.zeros(len(Xtr)), np.ones(len(Xg))]
            Xtr = pd.concat([Xtr, Xg], ignore_index=True).assign(source_gefs=src)
            ytr = np.r_[ytr, yg]
            wtr = np.r_[wtr, np.full(len(yg), v.get("gefs_weight", 0.3))]
            cols_used = cols + ["source_gefs"]
        elif v["gefs"] == "pretrain":
            init = lgb.train(lgb_params(cfg), lgb.Dataset(Xg, yg), num_boost_round=v.get("pre_rounds", 300))
            cols_used = cols
        else:
            raise ValueError(v["gefs"])
    else:
        cols_used = cols
    Xca = X.loc[ca, cols].assign(source_gefs=0.0) if "source_gefs" in cols_used else X.loc[ca, cols]
    Xev = X.loc[ev, cols].assign(source_gefs=0.0) if "source_gefs" in cols_used else X.loc[ev, cols]

    t0 = time.time()
    if v.get("per_lead_band"):
        bands = {"1-3": [1, 2, 3], "4-10": list(range(4, 11))}
        raw_ca, raw_ev = np.zeros(ca.sum()), np.zeros(ev.sum())
        for _, leads in bands.items():
            a = np.isin(Xtr["lead_day"], leads); b = np.isin(Xca["lead_day"], leads); c = np.isin(Xev["lead_day"], leads)
            bst = lgb.train(params, lgb.Dataset(Xtr[a], ytr[a], weight=wtr[a]), num_boost_round=3000,
                            valid_sets=[lgb.Dataset(Xca[b], y[ca][b])],
                            callbacks=[lgb.early_stopping(150, verbose=False)])
            raw_ca[b] = bst.predict(Xca[b], num_iteration=bst.best_iteration)
            raw_ev[c] = bst.predict(Xev[c], num_iteration=bst.best_iteration)
        n_trees = None
    else:
        bst = lgb.train(params, lgb.Dataset(Xtr, ytr, weight=wtr), num_boost_round=3000,
                        valid_sets=[lgb.Dataset(Xca, y[ca])], init_model=init,
                        callbacks=[lgb.early_stopping(150, verbose=False)])
        raw_ca = bst.predict(Xca, num_iteration=bst.best_iteration)
        raw_ev = bst.predict(Xev, num_iteration=bst.best_iteration)
        n_trees = bst.best_iteration

    # calibration on the calibration year (optionally per cluster)
    codes_ca = df.loc[ca, "subdivision_code"].to_numpy()
    codes_ev = df.loc[ev, "subdivision_code"].to_numpy()
    if v.get("cluster_calib"):
        cl = lambda c: np.isin(c, list(meta["event_sets"]["COAST"]))  # noqa: E731
        p_ev = np.zeros(ev.sum())
        for flag in (True, False):
            a, b = cl(codes_ca) == flag, cl(codes_ev) == flag
            iso = Isotonic().fit(raw_ca[a], y[ca][a])
            p_ev[b] = iso(raw_ev[b])
    else:
        p_ev = Isotonic().fit(raw_ca, y[ca])(raw_ev)
    p_ev = np.clip(p_ev, 0.01, 0.99)
    out = pd.DataFrame({"p": p_ev, "y": y[ev], "trigger": lab.loc[ev, "trigger_reason"].to_numpy(),
                        "lead": df.loc[ev, "lead_day"].to_numpy(), "code": codes_ev,
                        "date": df.loc[ev, "date"].to_numpy()})
    if v.get("mag_head"):
        ym = mag.astype(int)
        hm = lgb.train(params, lgb.Dataset(X.loc[tr, cols], ym[tr]), num_boost_round=3000,
                       valid_sets=[lgb.Dataset(X.loc[ca, cols], ym[ca])],
                       callbacks=[lgb.early_stopping(150, verbose=False)])
        iso_m = Isotonic().fit(hm.predict(X.loc[ca, cols], num_iteration=hm.best_iteration), ym[ca])
        pm_ca = iso_m(hm.predict(X.loc[ca, cols], num_iteration=hm.best_iteration))
        pm_ev = iso_m(hm.predict(X.loc[ev, cols], num_iteration=hm.best_iteration))
        # smallest threshold on the CALIBRATION year that keeps the head's precision >= its target
        target = v["mag_head"]
        order = np.sort(np.unique(pm_ca))[::-1]
        t_sel = order[-1]
        for t in order:
            f = pm_ca >= t
            if f.sum() >= 5 and ym[ca][f].mean() < target:
                break
            t_sel = t
        out["p_mag"] = pm_ev
        out["mag_flag"] = pm_ev >= t_sel
        out.attrs["mag_threshold"] = float(t_sel)
    return out, {"trees": n_trees, "seconds": round(time.time() - t0, 1), "n_features": len(cols_used)}


def summarize(o, flag_rate=None):
    busts = o["y"] == 1
    mag = o["trigger"].isin(["magnitude", "both"])
    flag = o["p"] >= ALERT
    r = {"pr_auc": fast_ap(o["y"], o["p"]),
         "pr_auc_lead1_3": fast_ap(o.loc[o["lead"] <= 3, "y"], o.loc[o["lead"] <= 3, "p"]),
         "pr_auc_lead4_10": fast_ap(o.loc[o["lead"] > 3, "y"], o.loc[o["lead"] > 3, "p"]),
         "precision_alert": float(o.loc[flag, "y"].mean()) if flag.any() else float("nan"),
         "recall_alert": float(flag[busts].mean()),
         "mag_recall_alert": float(flag[mag].mean()),
         "flag_rate": float(flag.mean()),
         "pr_auc_coastal_weak": fast_ap(o.loc[o["code"].isin(COASTAL_WEAK), "y"],
                                        o.loc[o["code"].isin(COASTAL_WEAK), "p"]),
         "pr_auc_other": fast_ap(o.loc[~o["code"].isin(COASTAL_WEAK), "y"],
                                 o.loc[~o["code"].isin(COASTAL_WEAK), "p"])}
    if "mag_flag" in o:  # big-miss head: extra flags on top of the unchanged main alert
        both = flag | o["mag_flag"]
        r["mag_head_pr_auc"] = fast_ap(mag, o["p_mag"])
        r["mag_recall_alert_or_head"] = float(both[mag].mean())
        r["extra_flag_rate"] = float((o["mag_flag"] & ~flag).mean())
        r["head_precision_for_mag"] = float(mag[o["mag_flag"]].mean()) if o["mag_flag"].any() else float("nan")
        r["mag_threshold"] = o.attrs.get("mag_threshold")
    if flag_rate is not None:  # magnitude recall at the SAME review budget as the baseline
        k = int(round(flag_rate * len(o)))
        top = np.argsort(-o["p"].to_numpy(), kind="stable")[:k]
        sel = np.zeros(len(o), bool); sel[top] = True
        r["mag_recall_same_budget"] = float(sel[mag.to_numpy()].mean())
        r["recall_same_budget"] = float(sel[busts.to_numpy()].mean())
    for L in range(1, 11):
        s = o["lead"] == L
        r[f"pr_auc_L{L}"] = fast_ap(o.loc[s, "y"], o.loc[s, "p"])
    return r


# ---------------------------------------------------------------- variants

VARIANTS = {
    "base": {},
    # Days 1-3 / feature poverty
    "lead_state": {"groups": ["lead_state"]},
    "antecedent": {"groups": ["antecedent"]},
    "interactions": {"groups": ["interactions"]},
    "feat_all": {"groups": ["lead_state", "antecedent", "interactions"]},
    "per_lead_band": {"per_lead_band": True},
    "feat_all_band": {"groups": ["lead_state", "antecedent", "interactions"], "per_lead_band": True},
    # magnitude busts
    "mag_w3": {"mag_weight": 3},
    "mag_w6": {"mag_weight": 6},
    "mag_w12": {"mag_weight": 12},
    "pos_w2": {"pos_weight": 2},
    "mag_w6_pos_w2": {"mag_weight": 6, "pos_weight": 2},
    # coastal
    "static": {"groups": ["static"]},
    "cluster_calib": {"cluster_calib": True},
    "static_cluster": {"groups": ["static"], "cluster_calib": True},
    # GEFS as auxiliary signal (years >= 2016 only: no overlap with NCMRWF observation days)
    "gefs_pool_w01": {"gefs": "pool", "gefs_weight": 0.1},
    "gefs_pool_w03": {"gefs": "pool", "gefs_weight": 0.3},
    "gefs_pool_w1": {"gefs": "pool", "gefs_weight": 1.0},
    "gefs_pretrain": {"gefs": "pretrain", "pre_rounds": 200},
    # round 2 (informed by round 1 on validation)
    "combo_band_gefs": {"groups": ["lead_state", "antecedent", "interactions"], "per_lead_band": True,
                        "gefs": "pool", "gefs_weight": 0.3},
    "combo_feat_gefs": {"groups": ["antecedent", "interactions"], "gefs": "pool", "gefs_weight": 0.3},
    "combo_feat_static_gefs": {"groups": ["antecedent", "interactions", "static"], "gefs": "pool", "gefs_weight": 0.3},
    "coastal_w2": {"coastal_weight": 2},
    "coastal_w2_gefs": {"coastal_weight": 2, "gefs": "pool", "gefs_weight": 0.3},
    "static_cluster_gefs": {"groups": ["static"], "cluster_calib": True, "gefs": "pool", "gefs_weight": 0.3},
    "all_static_cluster_gefs": {"groups": ["antecedent", "interactions", "static"], "cluster_calib": True,
                                "gefs": "pool", "gefs_weight": 0.3},
    "mag_head_p15": {"mag_head": 0.15},
    "mag_head_p25": {"mag_head": 0.25},
}


def gefs_block(meta, cols_from):
    """GEFS rows as auxiliary training data: own thresholds (fit on GEFS 2016-2018), own
    feature builder, labels never mixed with NCMRWF thresholds. 2015 is excluded so no GEFS row
    shares an observation day with the NCMRWF test years."""
    gcfg = load_config(source="gefs")
    g = load_pairs(gcfg["data_path"])
    g = g[g["date"].dt.year >= 2016].reset_index(drop=True)
    gtr = (g["date"].dt.year <= 2018).to_numpy()
    thr = fit_thresholds(g[gtr], gcfg["bust"]["percentile"], gcfg["bust"]["min_error_mm"])
    y = label_frame(g, thr, gcfg["bust"])["is_bust"].to_numpy().astype(int)
    b = FeatureBuilder(meta, gcfg["bust"], "issue").fit(g[gtr], y[gtr], thr)
    X = b.transform(g, oof=np.ones(len(g), bool))
    # IMD history for the antecedent-rain features, from GEFS's own observed column (daily issues,
    # so every IMD day appears as some row's valid date)
    obs = (g.assign(date=g["date"] + pd.to_timedelta(g["lead_day"], "D"))
           .groupby(["subdivision_code", "date"], as_index=False)["observed_rain"].first()
           .rename(columns={"observed_rain": "observed_rain_mm"}))
    return {"X": X, "y": y, "df": g, "imd": obs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["select", "final"])
    ap.add_argument("--variants", default=",".join(VARIANTS))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg, df = load()
    meta = load_meta(cfg)
    years = SELECT if a.mode == "select" else FINAL
    if a.mode == "select":
        df = df[df["date"].dt.year <= 2012].reset_index(drop=True)  # test years never enter selection
    names = a.variants.split(",")
    gefs = gefs_block(meta, None) if any(VARIANTS[n].get("gefs") for n in names) else None
    rows, preds = [], {}
    base_rate = None
    for n in names:
        o, info = run(cfg, meta, df, years, VARIANTS[n], gefs)
        if n == "base":
            base_rate = float((o["p"] >= ALERT).mean())
        r = summarize(o, base_rate)
        r.update(variant=n, **info)
        rows.append(r)
        preds[n] = o
        print(f"{n:16s} PR-AUC {r['pr_auc']:.3f}  L1-3 {r['pr_auc_lead1_3']:.3f}  L4-10 {r['pr_auc_lead4_10']:.3f}  "
              f"magRec@alert {r['mag_recall_alert']:.3f}  magRec@budget {r.get('mag_recall_same_budget', float('nan')):.3f}  "
              f"coastal {r['pr_auc_coastal_weak']:.3f}  other {r['pr_auc_other']:.3f}  ({info['seconds']}s)", flush=True)
    res = pd.DataFrame(rows).set_index("variant")
    if "base" in preds:
        o0 = preds["base"]
        for n in names:
            if n == "base":
                continue
            o = preds[n]
            for grp, mask in (("all", np.ones(len(o0), bool)), ("L1_3", (o0["lead"] <= 3).to_numpy()),
                              ("coastal", o0["code"].isin(COASTAL_WEAK).to_numpy())):
                b = block_bootstrap(o0["y"][mask], {"v": o["p"][mask], "b": o0["p"][mask]}, o0["date"][mask], reps=300)
                d = b["differences"]["v - b"]
                res.loc[n, f"diff_{grp}"] = d["mean"]
                res.loc[n, f"diff_{grp}_lo"] = d["lo"]
                res.loc[n, f"diff_{grp}_hi"] = d["hi"]
    res.to_csv(OUT / f"{a.mode}.csv")
    if a.mode == "final" and "base" in preds:
        o0 = preds["base"]
        boot = {}
        for n in names:
            if n == "base":
                continue
            o = preds[n]
            b = block_bootstrap(o0["y"], {n: o["p"], "base": o0["p"]}, o0["date"], reps=500)
            d = b["differences"][f"{n} - base"]
            boot[n] = {"pr_auc": b["pr_auc"][n], "diff_vs_base": d}
            for grp, mask in (("lead1_3", o0["lead"] <= 3), ("coastal_weak", o0["code"].isin(COASTAL_WEAK))):
                bb = block_bootstrap(o0.loc[mask, "y"], {n: o.loc[mask, "p"], "base": o0.loc[mask, "p"]},
                                     o0.loc[mask, "date"], reps=500)
                boot[n][f"diff_{grp}"] = bb["differences"][f"{n} - base"]
        (OUT / "final_bootstrap.json").write_text(json.dumps(boot, indent=2, default=float))
        print(json.dumps(boot, indent=1, default=float))


if __name__ == "__main__":
    main()
