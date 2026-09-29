"""Feature engineering. Every feature is known when the forecast is issued.

Inputs used: forecast_rain (this and earlier runs), ERA5 state at the issue date, and
tables fitted on training years (climatology, bust thresholds, historical bust rates).
Optional: forecast_rain_spread (ensemble std across members, e.g. NCMRWF S2S); missing
ERA5 variables simply become NaN features, which LightGBM handles.
Never used: observed_rain, error, is_bust, trigger_reason of the row or of any row whose
valid date is on or after the issue date.

Z-scores are taken per (subdivision, calendar month) so the model is indifferent to the
ERA5 units the data team ships (Pa or hPa, K or degC, m or mm).
"""
import numpy as np
import pandas as pd

from .schema import ERA5

FEATURE_GROUPS = {
    "lead_day": "lead",
    "forecast_rain": "amount", "fc_log1p": "amount", "fc_clim_ratio": "amount",
    "fc_minus_rain_threshold": "amount",
    "fc_jump": "run_change", "fc_jump_abs": "run_change", "fc_jump_gap": "run_change",
    "fc_neighbor_std": "lead_consistency",
    "fc_spread": "ensemble", "fc_spread_rel": "ensemble",
    "fc_region_mean": "regional", "fc_region_dev": "regional",
    "mslp_z": "pressure", "surface_pressure_z": "pressure", "mslp_region_z": "pressure", "mslp_tendency": "pressure",
    "dewpoint_z": "moisture", "dewpoint_depression": "moisture", "dewpoint_depression_z": "moisture",
    "temp_z": "temperature",
    "wind_u_z": "wind", "wind_v_z": "wind", "wind_speed_z": "wind",
    "era5_tp_rel": "recent_rain",
    "hist_bust_rate": "history", "bust_threshold_mm": "history", "clim_wet_frac": "history",
    "month_sin": "season", "month_cos": "season", "region_id": "season",
}
FEATURES = list(FEATURE_GROUPS)

# ERA5-derived state per (issue date, subdivision); z-scored per (subdivision, month).
_STATE_Z = {"mslp": "mslp_z", "surface_pressure": "surface_pressure_z", "dewpoint_2m": "dewpoint_z",
            "temp_2m": "temp_z", "wind_u10": "wind_u_z", "wind_v10": "wind_v_z",
            "wind_speed": "wind_speed_z", "dewpoint_depression": "dewpoint_depression_z"}
_SMOOTH = 20.0  # pseudo-count shrinking sparse bust-rate cells toward the lead-day rate


def issue_state(df, timing, shift_days=0):
    """Raw ERA5 values at each issue date, one row per (date, subdivision_code).

    timing='issue': the table already holds issue-date values (constant across leads).
    timing='valid': the table holds valid-date values; the state at issue date d is read
    from any row whose valid date is d, i.e. from an earlier issue. Issue dates with no
    such row get NaN (the model handles missing values)."""
    if timing == "issue":
        st = df.groupby(["date", "subdivision_code"], sort=False)[ERA5].first()
    elif timing == "valid":
        v = df.assign(date=df["date"] + pd.to_timedelta(df["lead_day"], "D"))
        st = v.groupby(["date", "subdivision_code"], sort=False)[ERA5].first()
        keys = pd.MultiIndex.from_frame(df[["date", "subdivision_code"]].drop_duplicates())
        st = st.reindex(keys)
    else:
        raise ValueError(f"era5 timing must be 'issue' or 'valid', got {timing!r}")
    st = st.reset_index()
    if shift_days:
        # Use the state from `shift_days` earlier: when the table's ERA5 values are daily means /
        # sums over the issue date, they include hours after a 00Z issue (the Day-1 window).
        prev = st.assign(date=st["date"] + pd.Timedelta(days=shift_days))
        st = st[["date", "subdivision_code"]].merge(prev, on=["date", "subdivision_code"], how="left")
    st["wind_speed"] = np.hypot(st["wind_u10"], st["wind_v10"])
    st["dewpoint_depression"] = st["temp_2m"] - st["dewpoint_2m"]
    return st


def _valid(df):
    return df["date"] + pd.to_timedelta(df["lead_day"], "D")


def _bustrate_table(keys, y):
    """Smoothed P(bust | subdivision, lead, valid month) and the per-lead fallback."""
    k = keys.assign(y=np.asarray(y, float))
    lead_rate = k.groupby("lead_day")["y"].mean()
    t = k.groupby(["subdivision_code", "lead_day", "vmonth"])["y"].agg(["sum", "count"]).reset_index()
    prior = t["lead_day"].map(lead_rate)
    t["rate"] = (t["sum"] + _SMOOTH * prior) / (t["count"] + _SMOOTH)
    return t[["subdivision_code", "lead_day", "vmonth", "rate"]], lead_rate


def _attach_rate(keys, table, lead_rate):
    r = keys.merge(table, on=["subdivision_code", "lead_day", "vmonth"], how="left")["rate"]
    return r.fillna(keys["lead_day"].map(lead_rate)).fillna(lead_rate.mean()).to_numpy(float, copy=True)


class FeatureBuilder:
    def __init__(self, meta, bust_cfg, era5_timing="issue", era5_shift_days=0):
        self.region_of = {s["code"]: s["region_key"] for s in meta["subdivisions"]}
        self.region_ids = {r["key"]: i for i, r in enumerate(meta["regions"])}
        self.rain_thr = bust_cfg["rain_threshold_mm"]
        self.era5_timing = era5_timing
        self.era5_shift_days = int(era5_shift_days)
        self.fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, train, y_train, thresholds):
        """train: training-year rows (with observed_rain); y_train: their bust labels;
        thresholds: output of labels.fit_thresholds on the same rows."""
        st = issue_state(train, self.era5_timing, self.era5_shift_days)
        st["month"] = st["date"].dt.month
        cols = list(_STATE_Z)
        g = st.groupby(["subdivision_code", "month"])[cols]
        self.state_mean = g.mean()
        self.state_std = g.std().clip(lower=1e-6)
        # Precipitation is skewed: store the mean for a ratio instead of a z-score.
        self.tp_mean = st.groupby(["subdivision_code", "month"])["total_precipitation"].mean()

        vd = train.assign(valid=_valid(train)).drop_duplicates(["subdivision_code", "valid"])
        vd = vd.assign(vmonth=vd["valid"].dt.month, wet=vd["observed_rain"] >= self.rain_thr)
        clim = vd.groupby(["subdivision_code", "vmonth"]).agg(clim_obs_mean=("observed_rain", "mean"),
                                                               clim_wet_frac=("wet", "mean"))
        self.clim = clim
        self.thresholds = thresholds[["subdivision_code", "lead_day", "p_err", "tau"]].copy()

        keys = self._rate_keys(train)
        self.rate_table, self.lead_rate = _bustrate_table(keys, y_train)
        self.train_years = sorted(train["date"].dt.year.unique().tolist())
        self._oof = {}
        for yr in self.train_years:  # out-of-fold tables for the training rows themselves
            other = (train["date"].dt.year != yr).to_numpy()
            if other.any():
                self._oof[yr] = _bustrate_table(keys[other], np.asarray(y_train)[other])

        counts = pd.DataFrame({"subdivision_code": train["subdivision_code"].to_numpy(),
                               "lead_day": train["lead_day"].to_numpy(), "y": np.asarray(y_train, float)})
        self.support = counts.groupby(["subdivision_code", "lead_day"])["y"].agg(n="count", busts="sum").reset_index()
        self.fitted = True
        return self

    def _rate_keys(self, df):
        return pd.DataFrame({"subdivision_code": df["subdivision_code"].to_numpy(),
                             "lead_day": df["lead_day"].to_numpy(),
                             "vmonth": _valid(df).dt.month.to_numpy()})

    # ------------------------------------------------------------ transform
    def transform(self, df, oof=None):
        """Feature matrix aligned with df's rows. df may contain earlier issue dates
        (history) that later rows use for run-to-run change and pressure tendency.
        oof: boolean mask of training rows (or True for all rows). Those rows take their
        historical bust rate from the other training years, so a row's own label never
        feeds its own feature."""
        assert self.fitted, "call fit() first"
        df = df.reset_index(drop=True)
        X = pd.DataFrame(index=df.index)
        fc = df["forecast_rain"].to_numpy(float)
        valid = _valid(df)
        vmonth = valid.dt.month
        X["lead_day"] = df["lead_day"].to_numpy(float)
        X["forecast_rain"] = fc
        X["fc_log1p"] = np.log1p(fc)
        ck = pd.MultiIndex.from_arrays([df["subdivision_code"], vmonth])
        clim = self.clim.reindex(ck)
        X["fc_clim_ratio"] = np.log1p(fc) - np.log1p(clim["clim_obs_mean"].to_numpy())
        X["fc_minus_rain_threshold"] = fc - self.rain_thr
        # Ensemble spread across members (NaN when the forecast source is deterministic).
        spread = df["forecast_rain_spread"].to_numpy(float) if "forecast_rain_spread" in df else np.full(len(df), np.nan)
        X["fc_spread"] = spread
        X["fc_spread_rel"] = spread / (fc + 1.0)

        # run-to-run change: the most recent earlier issue that covers the same valid date
        k = pd.DataFrame({"s": df["subdivision_code"], "v": valid, "d": df["date"], "fc": fc})
        k = k.sort_values(["s", "v", "d"], kind="stable")
        g = k.groupby(["s", "v"], sort=False)
        prev_fc, prev_d = g["fc"].shift(1), g["d"].shift(1)
        X["fc_jump"] = (k["fc"] - prev_fc).reindex(df.index)
        X["fc_jump_abs"] = X["fc_jump"].abs()
        X["fc_jump_gap"] = (k["d"] - prev_d).dt.days.reindex(df.index).astype(float)

        # consistency of this run across neighbouring lead days
        k2 = pd.DataFrame({"d": df["date"], "s": df["subdivision_code"], "L": df["lead_day"], "fc": fc})
        k2 = k2.sort_values(["d", "s", "L"], kind="stable")
        g2 = k2.groupby(["d", "s"], sort=False)["fc"]
        trio = np.column_stack([g2.shift(1), k2["fc"], g2.shift(-1)])
        ok = np.isfinite(trio)
        n = ok.sum(1)
        mean = np.where(ok, trio, 0).sum(1) / np.maximum(n, 1)
        ss = np.where(ok, (trio - mean[:, None]) ** 2, 0).sum(1)
        nstd = np.where(n >= 2, np.sqrt(ss / np.maximum(n - 1, 1)), np.nan)
        X["fc_neighbor_std"] = pd.Series(nstd, index=k2.index).reindex(df.index)

        region = df["subdivision_code"].map(self.region_of)
        X["region_id"] = region.map(self.region_ids).astype(float)
        rm = pd.Series(fc).groupby([df["date"], df["lead_day"], region]).transform("mean")
        X["fc_region_mean"] = rm.to_numpy()
        X["fc_region_dev"] = fc - rm.to_numpy()

        # ERA5 state at issue date
        st = issue_state(df, self.era5_timing, self.era5_shift_days)
        st["month"] = st["date"].dt.month
        sk = pd.MultiIndex.from_arrays([st["subdivision_code"], st["month"]])
        mu, sd = self.state_mean.reindex(sk), self.state_std.reindex(sk)
        for raw, name in _STATE_Z.items():
            st[name] = (st[raw].to_numpy() - mu[raw].to_numpy()) / sd[raw].to_numpy()
        st["era5_tp_rel"] = np.log1p(st["total_precipitation"].to_numpy()
                                     / np.maximum(self.tp_mean.reindex(sk).to_numpy(), 1e-9))
        st["mslp_region_z"] = st.groupby(["date", st["subdivision_code"].map(self.region_of)])["mslp_z"].transform("mean")
        st = st.sort_values(["subdivision_code", "date"], kind="stable")
        gap = st.groupby("subdivision_code")["date"].diff().dt.days
        tend = st.groupby("subdivision_code")["mslp_z"].diff() / gap
        st["mslp_tendency"] = tend.where(gap <= 3)
        keep = list(_STATE_Z.values()) + ["era5_tp_rel", "mslp_region_z", "mslp_tendency", "dewpoint_depression"]
        m = df[["date", "subdivision_code"]].merge(st[["date", "subdivision_code"] + keep],
                                                   on=["date", "subdivision_code"], how="left")
        for c in keep:
            X[c] = m[c].to_numpy(float)

        # fitted history tables
        X["bust_threshold_mm"] = df[["subdivision_code", "lead_day"]].merge(
            self.thresholds, on=["subdivision_code", "lead_day"], how="left")["tau"].to_numpy(float)
        X["clim_wet_frac"] = clim["clim_wet_frac"].to_numpy(float)
        keys = self._rate_keys(df)
        rate = _attach_rate(keys, self.rate_table, self.lead_rate)
        if oof is not None and np.any(oof):
            mask = np.ones(len(df), bool) if oof is True else np.asarray(oof, bool)
            yr = df["date"].dt.year.to_numpy()
            for y, (tb, lr) in self._oof.items():
                own = (yr == y) & mask
                if own.any():
                    rate[own] = _attach_rate(keys[own], tb, lr)
        X["hist_bust_rate"] = rate

        doy = valid.dt.dayofyear.to_numpy(float)
        X["month_sin"] = np.sin(2 * np.pi * doy / 365.25)
        X["month_cos"] = np.cos(2 * np.pi * doy / 365.25)
        return X[FEATURES].astype("float32")

    # ------------------------------------------------------------ persistence
    def state_dict(self):
        def frame(f):
            return f.reset_index().to_dict(orient="list")
        return {
            "region_of": self.region_of, "region_ids": self.region_ids, "rain_thr": self.rain_thr,
            "era5_timing": self.era5_timing, "era5_shift_days": self.era5_shift_days, "train_years": self.train_years,
            "state_mean": frame(self.state_mean), "state_std": frame(self.state_std),
            "tp_mean": frame(self.tp_mean), "clim": frame(self.clim),
            "thresholds": self.thresholds.to_dict(orient="list"),
            "rate_table": self.rate_table.to_dict(orient="list"),
            "lead_rate": {int(k): float(v) for k, v in self.lead_rate.items()},
            "support": self.support.to_dict(orient="list"),
        }

    @classmethod
    def from_state(cls, d):
        self = cls.__new__(cls)
        self.region_of, self.region_ids, self.rain_thr = d["region_of"], d["region_ids"], d["rain_thr"]
        self.era5_timing, self.train_years = d["era5_timing"], d["train_years"]
        self.era5_shift_days = d.get("era5_shift_days", 0)
        idx = ["subdivision_code", "month"]
        self.state_mean = pd.DataFrame(d["state_mean"]).set_index(idx)
        self.state_std = pd.DataFrame(d["state_std"]).set_index(idx)
        self.tp_mean = pd.DataFrame(d["tp_mean"]).set_index(idx)["total_precipitation"]
        self.clim = pd.DataFrame(d["clim"]).set_index(["subdivision_code", "vmonth"])
        self.thresholds = pd.DataFrame(d["thresholds"])
        self.rate_table = pd.DataFrame(d["rate_table"])
        self.lead_rate = pd.Series({int(k): v for k, v in d["lead_rate"].items()})
        self.support = pd.DataFrame(d["support"])
        self._oof = {}
        self.fitted = True
        return self
