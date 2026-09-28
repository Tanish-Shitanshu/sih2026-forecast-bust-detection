"""Historical analog matcher: nearest past cases in a standardised feature space.

One index per lead day. Features are robust-standardised on the library, then weighted by
the model's global SHAP importance so "similar" means similar in what drives busts.
Cases from another region pay a distance penalty (half the typical neighbour distance), so
a close local precedent is preferred over a slightly closer one elsewhere.
A query issued on date d only sees library cases whose valid date is before d, so their
outcome was known at the time. The novelty reference uses the same past-only rule.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ANALOG_FEATURES = ["fc_log1p", "fc_clim_ratio", "fc_jump", "fc_neighbor_std", "fc_region_dev", "mslp_z",
                   "mslp_region_z", "dewpoint_z", "dewpoint_depression_z", "temp_z", "wind_speed_z",
                   "era5_tp_rel", "hist_bust_rate", "month_sin", "month_cos"]
META = ["date", "subdivision_code", "region", "lead_day", "forecast_rain", "observed_rain", "is_bust",
        "trigger_reason", "event"]
REGION_PENALTY = 0.5


def importance_weights(importance, cols=ANALOG_FEATURES):
    imp = np.array([importance.get(c, 0.0) for c in cols], float)
    if imp.sum() <= 0:
        return np.ones(len(cols))
    return np.clip(np.sqrt(imp / imp[imp > 0].mean()), 0.3, 2.0)


class AnalogIndex:
    def __init__(self, cols=ANALOG_FEATURES):
        self.cols = list(cols)

    def fit(self, X, meta, weights, ref_k=5, ref_sample=800, seed=0):
        """X: feature frame; meta: library rows with the META columns (labels known)."""
        A = X[self.cols].to_numpy(float)
        self.center = np.nanmedian(A, 0)
        q75, q25 = np.nanpercentile(A, [75, 25], 0)
        self.scale = np.where((q75 - q25) > 1e-9, q75 - q25, np.nanstd(A, 0) + 1e-9)
        self.w = np.asarray(weights, float)
        self.Z = self._z(A)
        self._set_meta(meta[META].reset_index(drop=True).copy())
        self._build()
        rng = np.random.default_rng(seed)
        # Typical neighbour distance per lead (sets the region penalty and similarity scale).
        self.scale_d = {}
        for L, (idx, nn) in self.nn.items():
            s = rng.choice(len(idx), min(500, len(idx)), replace=False)
            d, _ = nn.kneighbors(self.Z[idx[s]], n_neighbors=min(ref_k + 1, len(idx)))
            self.scale_d[L] = float(np.median(d[:, 1:]))
        # Novelty reference: past-only k-NN distances for library cases in the later half of
        # the record (earlier cases have too little past to compare against).
        self.ref_k, self.ref = ref_k, {}
        dates = self.meta["date"].to_numpy()
        cutoff = np.quantile(dates.astype("int64"), 0.5).astype("datetime64[ns]")
        for L, (idx, _) in self.nn.items():
            late = idx[dates[idx] >= cutoff]
            s = rng.choice(late, min(ref_sample, len(late)), replace=False)
            hits = self._query_z(self.Z[s], dates[s], np.full(len(s), L), self.meta["region"].to_numpy()[s], ref_k)
            self.ref[L] = np.sort([np.mean([d for _, d in h]) for h in hits if h])
        return self

    def _set_meta(self, meta):
        self.meta = meta
        self.meta["date"] = pd.to_datetime(self.meta["date"])
        self.meta["valid_date"] = self.meta["date"] + pd.to_timedelta(self.meta["lead_day"], "D")
        self._vdates = self.meta["valid_date"].to_numpy()
        self._regions = self.meta["region"].to_numpy()

    def _z(self, A):
        return np.nan_to_num((A - self.center) / self.scale, nan=0.0) * self.w

    def _build(self):
        self.nn = {}
        leads = self.meta["lead_day"].to_numpy()
        for L in np.unique(leads):
            idx = np.where(leads == L)[0]
            self.nn[int(L)] = (idx, NearestNeighbors(n_neighbors=10).fit(self.Z[idx]))

    def _query_z(self, Zq, dates, leads, regions, k):
        dates = np.asarray(pd.to_datetime(pd.Series(dates)).to_numpy())
        out = []
        for i in range(len(Zq)):
            L = int(leads[i])
            idx, nn = self.nn.get(L, (None, None))
            if idx is None:
                out.append([])
                continue
            pen = REGION_PENALTY * getattr(self, "scale_d", {}).get(L, 0.0)
            n = min(len(idx), max(60, k * 40))
            while True:
                d, j = nn.kneighbors(Zq[i:i + 1], n_neighbors=n)
                cand, d = idx[j[0]], d[0]
                ok = self._vdates[cand] < dates[i]
                cand, d = cand[ok], d[ok]
                if regions is not None and pen > 0:
                    rank = d + pen * (self._regions[cand] != regions[i])
                    o = np.argsort(rank, kind="stable")
                    cand, d = cand[o], d[o]
                if len(cand) >= k or n >= len(idx):
                    break
                n = min(len(idx), n * 4)
            out.append(list(zip(cand[:k].tolist(), d[:k].tolist())))
        return out

    def query(self, X, dates, leads, regions=None, k=5):
        """For each query row: [(library_index, distance), ...] of up to k analogs whose
        valid date is before the query's issue date, same-region cases preferred."""
        regions = None if regions is None else np.asarray(regions)
        return self._query_z(self._z(X[self.cols].to_numpy(float)), dates, np.asarray(leads, int), regions, k)

    def novelty(self, hit_lists, leads):
        """Percentile of the query's mean analog distance among history's own past-only
        analog distances (0.5 = typical, 0.99 = more unusual than 99% of history)."""
        out = np.full(len(hit_lists), np.nan)
        for i, (hits, L) in enumerate(zip(hit_lists, leads)):
            ref = self.ref.get(int(L))
            if ref is None or not len(ref) or not hits:
                continue
            m = np.mean([d for _, d in hits[: self.ref_k]])
            out[i] = np.searchsorted(ref, m) / len(ref)
        return out

    def similarity(self, dist, lead):
        return float(1.0 / (1.0 + dist / max(self.scale_d.get(int(lead), 1.0), 1e-9)))

    # ------------------------------------------------------------ persistence
    def save(self, d):
        d = Path(d)
        lib = self.meta.drop(columns="valid_date").copy()
        for j, c in enumerate(self.cols):
            lib["z_" + c] = self.Z[:, j].astype("float32")
        lib.to_parquet(d / "analogs.parquet", index=False)
        (d / "analogs.json").write_text(json.dumps({
            "cols": self.cols, "center": self.center.tolist(), "scale": self.scale.tolist(), "w": self.w.tolist(),
            "ref_k": self.ref_k, "scale_d": {str(k): v for k, v in self.scale_d.items()},
            "ref": {str(k): np.asarray(v).tolist() for k, v in self.ref.items()}}))

    @classmethod
    def load(cls, d):
        d = Path(d)
        p = json.loads((d / "analogs.json").read_text())
        self = cls(p["cols"])
        self.center, self.scale, self.w = (np.asarray(p[k], float) for k in ("center", "scale", "w"))
        self.ref_k = p["ref_k"]
        self.scale_d = {int(k): v for k, v in p["scale_d"].items()}
        self.ref = {int(k): np.asarray(v) for k, v in p["ref"].items()}
        lib = pd.read_parquet(d / "analogs.parquet")
        self.Z = lib[["z_" + c for c in self.cols]].to_numpy(float)
        self._set_meta(lib[META].copy())
        self._build()
        return self
