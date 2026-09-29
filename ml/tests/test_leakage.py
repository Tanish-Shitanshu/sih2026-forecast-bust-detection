"""The model may only see what a forecaster has when the forecast is issued."""
import numpy as np
import pandas as pd
import pytest

from vishwas_ml.features import FEATURES
from vishwas_ml.model import BustModel
from vishwas_ml.schema import OUTCOME_COLS


def test_no_outcome_column_is_a_feature():
    for c in OUTCOME_COLS:
        assert c not in FEATURES


def test_features_ignore_outcomes(bundle):
    b = BustModel.load(bundle["dir"]).builder
    df = bundle["data"]
    X1 = b.transform(df)
    rng = np.random.default_rng(0)
    scrambled = df.assign(observed_rain=rng.permutation(df["observed_rain"].to_numpy()),
                          error=0.0, is_bust=rng.random(len(df)) < 0.5, trigger_reason=None)
    X2 = b.transform(scrambled)
    pd.testing.assert_frame_equal(X1, X2)


def test_features_ignore_later_issues(bundle):
    """Changing or deleting forecasts issued after date d must not change d's features."""
    b = BustModel.load(bundle["dir"]).builder
    df = bundle["data"]
    d = sorted(df["date"].unique())[len(df["date"].unique()) // 2]
    X_full = b.transform(df)
    X_past = b.transform(df[df["date"] <= d])
    mask = (df["date"] == d).to_numpy()
    past_rows = (df[df["date"] <= d]["date"] == d).to_numpy()
    pd.testing.assert_frame_equal(X_full[mask].reset_index(drop=True), X_past[past_rows].reset_index(drop=True))


def test_training_rows_get_out_of_fold_bust_rate(bundle):
    from vishwas_ml.splits import year_split

    b = BustModel.load(bundle["dir"]).builder
    df = bundle["data"]
    split = year_split(df, bundle["cfg"]["split"])
    # The saved builder has no OOF tables, so refit one to compare in-sample vs OOF.
    from vishwas_ml.features import FeatureBuilder
    from vishwas_ml.config import load_meta
    from vishwas_ml.labels import fit_thresholds, label_frame
    cfg = bundle["cfg"]
    tr = split["train"]
    thr = fit_thresholds(df[tr], 95, 25.0)
    y = label_frame(df, thr, cfg["bust"])["is_bust"].to_numpy()
    fb = FeatureBuilder(load_meta(cfg), cfg["bust"], b.era5_timing).fit(df[tr], y[tr], thr)
    ins = fb.transform(df)["hist_bust_rate"].to_numpy()
    oof = fb.transform(df, oof=tr)["hist_bust_rate"].to_numpy()
    assert not np.allclose(ins[tr], oof[tr])           # training rows differ
    assert np.allclose(ins[~tr], oof[~tr])             # calibration/test rows untouched


def test_valid_time_era5_is_read_from_earlier_rows(small_data):
    """If the data team ships valid-date ERA5 values, issue-date state comes from rows whose
    valid date equals the issue date, never from the row's own (future) values."""
    from vishwas_ml.features import issue_state

    df = small_data.copy()
    st_issue = issue_state(df, "issue").set_index(["date", "subdivision_code"])
    # Build a valid-time version: row (d, L) carries the issue-state of date d+L.
    v = df[["date", "subdivision_code", "lead_day"]].copy()
    v["vd"] = v["date"] + pd.to_timedelta(v["lead_day"], "D")
    lookup = st_issue.reindex(pd.MultiIndex.from_arrays([v["vd"], v["subdivision_code"]]))
    vt = df.copy()
    for c in ["mslp", "temp_2m"]:
        vt[c] = lookup[c].to_numpy()
    vt = vt.dropna(subset=["mslp"])
    st_valid = issue_state(vt, "valid").set_index(["date", "subdivision_code"])
    common = st_valid.dropna(subset=["mslp"]).index.intersection(st_issue.index)
    assert len(common) > 0
    assert np.allclose(st_valid.loc[common, "mslp"], st_issue.loc[common, "mslp"])


@pytest.mark.parametrize("col", ["observed_rain", "is_bust"])
def test_predictions_ignore_outcomes(bundle, col):
    m = BustModel.load(bundle["dir"])
    df = bundle["data"]
    p1 = m.predict(m.builder.transform(df))
    p2 = m.predict(m.builder.transform(df.assign(**{col: df[col].sample(frac=1, random_state=0).to_numpy()})))
    assert np.array_equal(p1, p2)


def test_era5_shift_uses_previous_day_state(small_data):
    """era5_shift_days=1: the issue-date state is read from the day before (for tables whose ERA5
    columns are daily means/sums over the issue date, i.e. partly after a 00Z issue)."""
    from vishwas_ml.features import issue_state

    df = small_data[small_data["subdivision_code"] == "KL"]
    st0 = issue_state(df, "issue").set_index("date")["mslp"]
    st1 = issue_state(df, "issue", shift_days=1).set_index("date")["mslp"]
    d = st0.index[5]
    prev = d - pd.Timedelta(days=1)
    assert (np.isnan(st1.loc[d]) if prev not in st0.index else st1.loc[d] == st0.loc[prev])
    assert st1.loc[d] != st0.loc[d] or np.isnan(st1.loc[d])
