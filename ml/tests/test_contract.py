"""Response shapes must match the sample responses on the frontend's API page
(frontend/index.html, `var API`), so the backend can pass them through unchanged."""
import pandas as pd
import pytest

from vishwas_ml.config import load_meta

CONFIDENCE_MAP = {"cycle", "lead_day", "count", "items"}
CONFIDENCE_ITEM = {"code", "name", "bust_probability", "forecast_confidence", "level", "weather_event"}
BUST_PROB = {"subdivision", "lead_day", "bust_probability", "forecast_confidence", "level", "band", "weather_event",
             "recommended_action", "model_self_confidence"}
EXPLAIN = {"subdivision", "lead_day", "factors", "closest_analog"}
FACTOR = {"feature", "contribution_percent", "direction"}
ANALOG = {"date", "description", "outcome"}
TRUST = {"lead_day", "items"}
TRUST_ITEM = {"code", "model_self_confidence", "level", "reason"}
OUTCOME_RES = {"id", "subdivision", "lead_day", "predicted_bust_probability", "outcome", "status", "submitted_by"}


@pytest.fixture(scope="module")
def cycle(service):
    return service.cycles()[-5]


def test_confidence_map(service, cycle):
    meta = load_meta(service.cfg)
    r = service.confidence_map(3, cycle)
    assert set(r) == CONFIDENCE_MAP and r["count"] == 33 and len(r["items"]) == 33
    assert r["cycle"] == f"{pd.Timestamp(cycle):%Y-%m-%d}T00Z"
    assert [i["code"] for i in r["items"]] == [s["code"] for s in meta["subdivisions"]]
    tiers = {t["key"]: t for t in meta["tiers"]}
    for it in r["items"]:
        assert set(it) == CONFIDENCE_ITEM
        assert 0 < it["bust_probability"] < 1
        assert round(it["bust_probability"] + it["forecast_confidence"], 2) == 1.0
        assert it["level"] in tiers
        assert (it["weather_event"] is None) == (it["level"] == "green") or it["weather_event"] is None
        assert it["weather_event"] is None or it["weather_event"] in meta["events"]


def test_bust_probability_and_action_agree_with_tiers(service, cycle):
    meta = load_meta(service.cfg)
    for L in (1, 5, 10):
        r = service.bust_probability("KL", L, cycle)
        assert set(r) == BUST_PROB
        t = next(t for t in meta["tiers"] if t["key"] == r["level"])
        assert r["band"] == t["band"] and r["recommended_action"] == t["action"]
        assert 0 <= r["model_self_confidence"] <= 1
        a = service.action("KL", L, cycle)
        assert a["recommended_action"] == r["recommended_action"] and a["level"] == r["level"]


def test_explain(service, cycle):
    r = service.explain("W.UP", 4, cycle)
    assert set(r) == EXPLAIN
    assert len(r["factors"]) == 3
    for f in r["factors"]:
        assert set(f) == FACTOR and f["direction"] in ("raises", "lowers")
        assert isinstance(f["contribution_percent"], int) and 0 <= f["contribution_percent"] <= 100
        assert isinstance(f["feature"], str) and "nan" not in f["feature"].lower()
    assert r["closest_analog"] is None or set(r["closest_analog"]) == ANALOG


def test_model_trust(service, cycle):
    meta = load_meta(service.cfg)
    levels = {lv["label"].lower() for lv in meta["trust_levels"]}
    r = service.model_trust(lead_day=2, cycle=cycle)
    assert set(r) == TRUST and len(r["items"]) == 33
    conf = [i["model_self_confidence"] for i in r["items"]]
    assert conf == sorted(conf)  # least confident first
    for it in r["items"]:
        assert set(it) == TRUST_ITEM and it["level"] in levels
    per_sub = service.model_trust(subdivision="Kerala", cycle=cycle)
    assert per_sub["subdivision"] == "KL" and [i["lead_day"] for i in per_sub["items"]] == list(range(1, 11))


def test_analogs_only_use_known_outcomes(service, cycle):
    r = service.analogs("ASM", 3, cycle)
    assert r["count"] == len(r["items"]) > 0
    for it in r["items"]:
        valid = pd.Timestamp(it["date"]) + pd.Timedelta(days=it["lead_day"])
        assert valid < pd.Timestamp(cycle)
        assert 0 < it["similarity"] <= 1


def test_record_outcome_and_errors(service, cycle):
    r = service.record_outcome("W.UP", 3, "incorrect", note="Heavy spell arrived a day late", cycle=cycle)
    assert set(r) == OUTCOME_RES and r["status"] == "pending"
    r2 = service.record_outcome("W.UP", 3, "correct", cycle=cycle)
    assert r2["id"] == r["id"] + 1
    with pytest.raises(ValueError):
        service.record_outcome("W.UP", 3, "maybe")
    with pytest.raises(ValueError):
        service.bust_probability("XX", 3)
    with pytest.raises(ValueError):
        service.bust_probability("KL", 11)
    with pytest.raises(LookupError):
        service.confidence_map(3, "1999-01-01")


def test_big_miss_endpoint(service, cycle):
    r = service.big_miss(3, cycle)
    assert r["count"] == 33 and set(r["items"][0]) == {"code", "name", "big_miss_probability", "flagged"}
    a = service.action("KL", 3, cycle)
    assert "big_miss_watch" in a and set(a["big_miss_watch"]) == {"probability", "flagged", "note"}
    # contracted shapes are unchanged by the new head
    assert set(service.bust_probability("KL", 3, cycle)) == BUST_PROB
