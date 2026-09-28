import numpy as np
import pandas as pd

from vishwas_ml.labels import attach_tau, bust_labels, fit_thresholds


def test_locked_definition_cases():
    tau = np.full(6, 30.0)
    f = np.array([40.0, 0.0, 3.0, 2.0, 100.0, 2.4])
    o = np.array([5.0, 0.0, 1.0, 2.4, 10.0, 2.6])
    y, r = bust_labels(f, o, tau, rain_threshold_mm=2.5)
    # 40 vs 5: |35| >= 30 -> magnitude, both wet -> "magnitude"
    # 0 vs 0: nothing
    # 3 vs 1: rain vs no-rain -> category
    # 2 vs 2.4: both below 2.5, small error -> nothing
    # 100 vs 10: magnitude, both wet
    # 2.4 vs 2.6: flips across 2.5 -> category
    assert y.tolist() == [True, False, True, False, True, True]
    assert r.tolist() == ["magnitude", None, "category", None, "magnitude", "category"]
    y2, r2 = bust_labels([50.0], [0.0], [25.0])
    assert y2[0] and r2[0] == "both"


def test_threshold_is_exactly_at_boundary():
    y, r = bust_labels([35.0], [10.0], [25.0])  # |error| == tau counts (>=)
    assert y[0] and r[0] == "magnitude"


def test_floor_and_percentile_per_subdivision_lead():
    rng = np.random.default_rng(0)
    rows = []
    for code, scale in (("KL", 40.0), ("W.RJ", 2.0)):
        for L in (1, 5):
            e = rng.exponential(scale * L / 3, 400)
            rows.append(pd.DataFrame({"subdivision_code": code, "lead_day": L, "forecast_rain": e, "observed_rain": 0.0}))
    df = pd.concat(rows, ignore_index=True)
    thr = fit_thresholds(df, 95, 25.0).set_index(["subdivision_code", "lead_day"])
    assert thr.loc[("W.RJ", 1), "tau"] == 25.0  # dry subdivision: floor wins
    assert thr.loc[("KL", 5), "tau"] > thr.loc[("KL", 1), "tau"] > 25.0  # per lead day
    q = np.quantile(df.query("subdivision_code == 'KL' and lead_day == 5")["forecast_rain"], 0.95)
    assert np.isclose(thr.loc[("KL", 5), "p_err"], q)


def test_unseen_pair_falls_back_to_lead_median_not_own_data():
    thr = pd.DataFrame({"subdivision_code": ["KL", "TG"], "lead_day": [3, 3], "p_err": [40.0, 30.0],
                        "tau": [40.0, 30.0]})
    tau = attach_tau(pd.DataFrame({"subdivision_code": ["GJ"], "lead_day": [3]}), thr, 25.0)
    assert tau[0] == 35.0


def test_thresholds_use_training_years_only(bundle):
    """Thresholds saved in the bundle equal a fit on the training years alone."""
    from vishwas_ml.model import BustModel
    from vishwas_ml.splits import year_split

    df, cfg = bundle["data"], bundle["cfg"]
    split = year_split(df, cfg["split"])
    expect = fit_thresholds(df[split["train"]], cfg["bust"]["percentile"], cfg["bust"]["min_error_mm"])
    got = BustModel.load(bundle["dir"]).builder.thresholds
    m = expect.merge(got, on=["subdivision_code", "lead_day"], suffixes=("_e", "_g"))
    assert len(m) == len(expect) and np.allclose(m["tau_e"], m["tau_g"])
