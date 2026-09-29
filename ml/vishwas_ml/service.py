"""What the backend calls. One method per ML-backed endpoint; each returns a dict
matching the sample response on the frontend's API page (frontend/index.html, `var API`).

    svc = VishwasService()                   # loads ml/models + the pairs table
    svc.confidence_map(lead_day=3)           # GET /api/v1/confidence-map
    svc.bust_probability("W.UP", 3)          # GET /api/v1/bust-probability
    svc.explain("W.UP", 3)                   # GET /api/v1/subdivisions/{code}/explain
    svc.model_trust(lead_day=3)              # GET /api/v1/model-trust
    svc.analogs("W.UP", 3)                   # extra: top similar past events
    svc.action("W.UP", 3)                    # extra: rules-engine response tier
    svc.record_outcome("W.UP", 3, "incorrect", note="...")   # POST /api/v1/outcomes (dev store)

`cycle` is the forecast issue date ("2026-09-28", "2026-09-28T00Z" or a date); omitted =
latest cycle in the history table. Bad input raises ValueError (HTTP 422); an unknown
cycle raises LookupError (HTTP 404).
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .actions import as_pct, tier_for, trust_level, trust_note
from .analogs import AnalogIndex
from .config import load_config, load_meta
from .events import tag_events
from .explain import explain_rows
from .feedback import OUTCOMES, append_outcome, read_outcomes
from .model import BustModel
from .schema import ERA5, coerce, load_pairs
from .trust import REASONS, combine, flags

HISTORY_DAYS = 11  # earlier issues needed for run-to-run change, tendency and valid-time ERA5


def _none(v):
    return v if isinstance(v, str) else None


def _fmt_date(d):
    d = pd.Timestamp(d)
    return f"{d.day} {d.strftime('%b %Y')}"


class VishwasService:
    def __init__(self, models_dir=None, history=None, cfg=None, feedback_path=None):
        self.cfg = cfg or load_config()
        self.meta = load_meta(self.cfg)
        d = Path(models_dir or self.cfg["models_dir"])
        self.model = BustModel.load(d)
        self.analog_index = AnalogIndex.load(d)
        if history is None:
            history = load_pairs(self.cfg["data_path"])
        else:
            history = coerce(history.assign(**{c: history.get(c, np.nan) for c in
                                               ["observed_rain", "error", "is_bust", "trigger_reason"] + ERA5
                                               if c not in history}))
        self.history = history
        self.subs = {s["code"]: s for s in self.meta["subdivisions"]}
        self._by_name = {s["name"].lower(): s["code"] for s in self.meta["subdivisions"]}
        self.feedback_path = Path(feedback_path or self.cfg["models_dir"].parent / "data" / "feedback_dev.jsonl")
        self._cache = {}

    # ---------------------------------------------------------------- inputs
    def cycles(self):
        return sorted(self.history["date"].unique())

    def _cycle(self, cycle):
        if cycle is None:
            return pd.Timestamp(self.history["date"].max())
        s = str(cycle).replace("T00Z", "").replace("T00:00Z", "")
        try:
            c = pd.Timestamp(s).normalize()
        except ValueError as e:
            raise ValueError(f"cannot parse cycle {cycle!r}") from e
        if c not in set(self.history["date"]):
            raise LookupError(f"no forecast rows for cycle {c.date()}")
        return c

    def _code(self, subdivision):
        s = str(subdivision).strip()
        if s in self.subs:
            return s
        up = {c.upper(): c for c in self.subs}
        if s.upper() in up:
            return up[s.upper()]
        if s.lower() in self._by_name:
            return self._by_name[s.lower()]
        raise ValueError(f"unknown subdivision {subdivision!r}")

    @staticmethod
    def _lead(lead_day):
        try:
            L = int(lead_day)
        except (TypeError, ValueError) as e:
            raise ValueError("lead_day must be an integer from 1 to 10") from e
        if not 1 <= L <= 10:
            raise ValueError("lead_day must be from 1 to 10")
        return L

    # ---------------------------------------------------------------- scoring
    def score(self, cycle=None):
        """All subdivision x lead rows for one cycle, with probability, tier, event,
        trust and analogs. Cached per cycle."""
        c = self._cycle(cycle)
        if c in self._cache:
            return self._cache[c]
        h = self.history
        win = h[(h["date"] > c - pd.Timedelta(days=HISTORY_DAYS)) & (h["date"] <= c)].reset_index(drop=True)
        X_all = self.model.builder.transform(win)
        now = (win["date"] == c).to_numpy()
        rows = win[now].reset_index(drop=True)
        X = X_all[now].reset_index(drop=True)
        p = self.model.predict(X)
        contrib = self.model.contributions(X)
        vmonth = (rows["date"] + pd.to_timedelta(rows["lead_day"], "D")).dt.month.to_numpy()
        events = tag_events(X, rows["subdivision_code"], vmonth, self.meta["event_sets"])

        k, k_trust = self.cfg["analogs"]["k"], self.cfg["analogs"]["k_trust"]
        regions = rows["subdivision_code"].map(self.model.builder.region_of).to_numpy()
        hits = self.analog_index.query(X, rows["date"], rows["lead_day"], regions=regions, k=k_trust)
        lib = self.analog_index.meta
        analog_rate = np.array([lib["is_bust"].to_numpy()[[j for j, _ in hs]].mean() if hs else np.nan
                                for hs in hits])
        novelty = self.analog_index.novelty(hits, rows["lead_day"])
        sup = rows[["subdivision_code", "lead_day"]].merge(self.model.builder.support,
                                                           on=["subdivision_code", "lead_day"], how="left")
        f = flags(sup["busts"].fillna(0).to_numpy(), novelty, X, self.model.metadata["trust_ranges"], p,
                  analog_rate, self.cfg["trust"]["min_busts"])
        u, reason_key, active = combine(f)

        tiers = self.meta["tiers"]
        out = rows[["date", "subdivision_code", "lead_day", "forecast_rain"]].copy()
        out["p"] = p
        out["tier"] = [tier_for(x, tiers)["key"] for x in p]
        out["event"] = pd.Series([None if t == "green" else e for t, e in zip(out["tier"], events)], dtype=object)
        out["u"] = u
        out["trust_reason"] = [REASONS[r] for r in reason_key]
        out["trust_flags"] = active
        out["analog_bust_rate"] = analog_rate
        out["vmonth"] = vmonth
        res = {"rows": out, "X": X, "contrib": contrib, "hits": [hs[:k] for hs in hits], "cycle": c}
        self._cache[c] = res
        return res

    def _row(self, code, lead_day, cycle):
        code, L = self._code(code), self._lead(lead_day)
        sc = self.score(cycle)
        r = sc["rows"]
        i = np.where((r["subdivision_code"] == code) & (r["lead_day"] == L))[0]
        if not len(i):
            raise LookupError(f"no forecast for {code} day {L} in cycle {sc['cycle'].date()}")
        return sc, int(i[0])

    def _ctx(self, sc, i):
        r = sc["rows"].iloc[i]
        s = self.subs[r["subdivision_code"]]
        return {"month": int(r["vmonth"]), "rain_thr": self.cfg["bust"]["rain_threshold_mm"],
                "region_name": s["region_name"], "name": s["name"]}

    def _analog_items(self, sc, i, k=None):
        lib = self.analog_index.meta
        L = int(sc["rows"]["lead_day"].iloc[i])
        items = []
        for j, dist in sc["hits"][i][: k or len(sc["hits"][i])]:
            a = lib.iloc[j]
            s = self.subs.get(a["subdivision_code"], {"name": a["subdivision_code"]})
            fc, ob = float(a["forecast_rain"]), float(a["observed_rain"])
            ev = a["event"] if isinstance(a["event"], str) else None
            desc = f"{ev[0].upper() + ev[1:] if ev else 'Similar setup'}, {s['name']} (Day {int(a['lead_day'])} forecast)"
            if bool(a["is_bust"]):
                diff = fc - ob
                how = f"{'over' if diff > 0 else 'under'}-forecast by {abs(diff):.0f} mm"
                if a["trigger_reason"] == "category":
                    outcome = f"Busted: rain/no-rain call was wrong, forecast {fc:.1f} mm vs observed {ob:.1f} mm"
                else:
                    outcome = f"Busted: forecast {fc:.0f} mm vs observed {ob:.0f} mm, {how}"
            else:
                outcome = f"Did not bust: forecast {fc:.0f} mm vs observed {ob:.0f} mm"
            items.append({"date": _fmt_date(a["date"]), "subdivision": a["subdivision_code"],
                          "lead_day": int(a["lead_day"]), "similarity": round(self.analog_index.similarity(dist, L), 2),
                          "busted": bool(a["is_bust"]),
                          "trigger_reason": a["trigger_reason"] if bool(a["is_bust"]) else None,
                          "forecast_rain_mm": round(fc, 1), "observed_rain_mm": round(ob, 1),
                          "description": desc, "outcome": outcome})
        return items

    # ---------------------------------------------------------------- endpoints
    def confidence_map(self, lead_day, cycle=None):
        """GET /api/v1/confidence-map?cycle=&lead_day="""
        L = self._lead(lead_day)
        sc = self.score(cycle)
        r = sc["rows"][sc["rows"]["lead_day"] == L].set_index("subdivision_code")
        items = []
        for s in self.meta["subdivisions"]:
            if s["code"] not in r.index:
                continue
            x = r.loc[s["code"]]
            bp = as_pct(x["p"])
            items.append({"code": s["code"], "name": s["name"], "bust_probability": bp,
                          "forecast_confidence": round(1 - bp, 2), "level": x["tier"], "weather_event": _none(x["event"])})
        return {"cycle": f"{sc['cycle']:%Y-%m-%d}T00Z", "lead_day": L, "count": len(items), "items": items}

    def bust_probability(self, subdivision, lead_day, cycle=None):
        """GET /api/v1/bust-probability?subdivision=&lead_day="""
        sc, i = self._row(subdivision, lead_day, cycle)
        x = sc["rows"].iloc[i]
        t = tier_for(x["p"], self.meta["tiers"])
        bp = as_pct(x["p"])
        return {"subdivision": x["subdivision_code"], "lead_day": int(x["lead_day"]), "bust_probability": bp,
                "forecast_confidence": round(1 - bp, 2), "level": t["key"], "band": t["band"],
                "weather_event": _none(x["event"]), "recommended_action": t["action"],
                "model_self_confidence": as_pct(1 - x["u"])}

    def explain(self, code, lead_day, cycle=None, top=3):
        """GET /api/v1/subdivisions/{code}/explain?lead_day="""
        sc, i = self._row(code, lead_day, cycle)
        factors = explain_rows(sc["X"].iloc[[i]], sc["contrib"][[i]], [self._ctx(sc, i)], top=top)[0]
        an = self._analog_items(sc, i, k=1)
        closest = {k: an[0][k] for k in ("date", "description", "outcome")} if an else None
        x = sc["rows"].iloc[i]
        return {"subdivision": x["subdivision_code"], "lead_day": int(x["lead_day"]), "factors": factors,
                "closest_analog": closest}

    def analogs(self, subdivision, lead_day, cycle=None, k=None):
        """Top similar past events (the prompt's GET /analogs). Only cases whose outcome was
        known before this cycle are returned."""
        sc, i = self._row(subdivision, lead_day, cycle)
        x = sc["rows"].iloc[i]
        items = self._analog_items(sc, i, k=k or self.cfg["analogs"]["k"])
        br = x["analog_bust_rate"]
        return {"subdivision": x["subdivision_code"], "lead_day": int(x["lead_day"]),
                "cycle": f"{sc['cycle']:%Y-%m-%d}T00Z", "count": len(items),
                "analog_bust_rate": None if not np.isfinite(br) else round(float(br), 2), "items": items}

    def action(self, subdivision, lead_day, cycle=None):
        """Rules engine (the prompt's GET /action): probability band -> instruction."""
        b = self.bust_probability(subdivision, lead_day, cycle)
        conf_label = trust_level(1 - b["model_self_confidence"], self.meta["trust_levels"])
        return {"subdivision": b["subdivision"], "lead_day": b["lead_day"], "bust_probability": b["bust_probability"],
                "level": b["level"], "band": b["band"], "recommended_action": b["recommended_action"],
                "model_self_confidence": b["model_self_confidence"], "trust_note": trust_note(conf_label)}

    def model_trust(self, lead_day=None, subdivision=None, cycle=None, limit=None):
        """GET /api/v1/model-trust?lead_day=  -> least confident subdivisions first (frontend shape).
        With subdivision= (the prompt's ?region=): that subdivision across all lead days,
        with the individual reliability flags."""
        sc = self.score(cycle)
        r = sc["rows"]
        levels = self.meta["trust_levels"]

        def item(x, code_key="code"):
            return {code_key: x["subdivision_code"], "model_self_confidence": as_pct(1 - x["u"]),
                    "level": trust_level(x["u"], levels).lower(), "reason": x["trust_reason"]}

        if subdivision is not None:
            code = self._code(subdivision)
            rr = r[r["subdivision_code"] == code].sort_values("lead_day")
            return {"subdivision": code, "cycle": f"{sc['cycle']:%Y-%m-%d}T00Z",
                    "items": [dict(lead_day=int(x["lead_day"]), **{k: v for k, v in item(x).items() if k != "code"},
                                   flags=list(x["trust_flags"])) for _, x in rr.iterrows()]}
        L = self._lead(lead_day if lead_day is not None else 1)
        rr = r[r["lead_day"] == L].sort_values("u", ascending=False, kind="stable")
        items = [item(x) for _, x in rr.iterrows()]
        return {"lead_day": L, "items": items[:limit] if limit else items}

    def record_outcome(self, subdivision, lead_day, outcome, note=None, submitted_by="forecaster", cycle=None):
        """POST /api/v1/outcomes. Dev store only: the backend owns persistence and should call
        feedback.validate_outcome + its own DB, then pass approved rows to recalibration."""
        code, L = self._code(subdivision), self._lead(lead_day)
        if outcome not in OUTCOMES:
            raise ValueError("outcome must be correct, incorrect or partial")
        if note is not None and len(note) > 80:
            raise ValueError("note must be at most 80 characters")
        b = self.bust_probability(code, L, cycle)
        prev = read_outcomes(self.feedback_path)
        rec = {"id": (int(prev["id"].max()) + 1) if len(prev) else 1, "subdivision": code, "lead_day": L,
               "predicted_bust_probability": b["bust_probability"], "outcome": outcome, "status": "pending",
               "submitted_by": submitted_by}
        append_outcome(self.feedback_path, dict(rec, note=note, cycle=f"{self._cycle(cycle):%Y-%m-%d}",
                                                created_at=datetime.now(timezone.utc).isoformat(timespec="seconds")))
        return rec


def dumps(obj):
    return json.dumps(obj, indent=2, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
