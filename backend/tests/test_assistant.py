"""Tests for the Sarvam-backed assistant routes. Sarvam itself is mocked throughout (fast,
free, deterministic, no network) -- what's under test is the backend's own auth, error
handling, and grounding logic, not Sarvam's API. A real end-to-end call against the live
Sarvam API was made manually and is reported separately (not part of this suite, matching
how ml/tests never calls a live weather feed either).
"""
import sarvam
import service_registry
from conftest import CYCLE


# ---------- auth ----------

def test_transcribe_requires_auth(client):
    r = client.post("/api/v1/assistant/transcribe", files={"file": ("q.webm", b"fake-audio", "audio/webm")})
    assert r.status_code == 401


def test_chat_requires_auth(client):
    r = client.post("/api/v1/assistant/chat", json={"text": "what is Vishwas"})
    assert r.status_code == 401


def test_speak_requires_auth(client):
    r = client.post("/api/v1/assistant/speak", json={"text": "hello"})
    assert r.status_code == 401


# ---------- graceful failure when Sarvam is unreachable ----------

def test_chat_502_when_sarvam_unreachable(client, duty_h, monkeypatch):
    def boom(*a, **k):
        raise sarvam.SarvamError("simulated: could not reach Sarvam chat completions")
    monkeypatch.setattr(sarvam, "chat", boom)
    r = client.post("/api/v1/assistant/chat", json={"text": "what is Vishwas"}, headers=duty_h)
    assert r.status_code == 502
    assert "simulated" in r.json()["detail"]


def test_transcribe_502_when_sarvam_unreachable(client, duty_h, monkeypatch):
    def boom(*a, **k):
        raise sarvam.SarvamError("simulated: could not reach Sarvam speech-to-text")
    monkeypatch.setattr(sarvam, "transcribe", boom)
    r = client.post("/api/v1/assistant/transcribe",
                     files={"file": ("q.webm", b"fake-audio-bytes", "audio/webm")}, headers=duty_h)
    assert r.status_code == 502
    assert "simulated" in r.json()["detail"]


def test_speak_502_when_sarvam_unreachable(client, duty_h, monkeypatch):
    def boom(*a, **k):
        raise sarvam.SarvamError("simulated: could not reach Sarvam text-to-speech")
    monkeypatch.setattr(sarvam, "speak", boom)
    r = client.post("/api/v1/assistant/speak", json={"text": "hello"}, headers=duty_h)
    assert r.status_code == 502
    assert "simulated" in r.json()["detail"]


def test_transcribe_rejects_empty_audio(client, duty_h):
    r = client.post("/api/v1/assistant/transcribe", files={"file": ("q.webm", b"", "audio/webm")}, headers=duty_h)
    assert r.status_code == 422


# ---------- mocked success paths ----------

def test_transcribe_success(client, duty_h, monkeypatch):
    def fake_transcribe(audio_bytes, filename, content_type, language_code=None):
        assert audio_bytes == b"fake-audio-bytes"
        return {"transcript": "why is Tamil Nadu orange", "language_code": "en-IN", "language_probability": 0.98}
    monkeypatch.setattr(sarvam, "transcribe", fake_transcribe)
    r = client.post("/api/v1/assistant/transcribe",
                     files={"file": ("q.webm", b"fake-audio-bytes", "audio/webm")}, headers=duty_h)
    assert r.status_code == 200
    body = r.json()
    assert body["transcript"] == "why is Tamil Nadu orange"
    assert body["language_code"] == "en-IN"


def test_speak_success(client, duty_h, monkeypatch):
    monkeypatch.setattr(sarvam, "speak", lambda text, language_code="en-IN", speaker=None: ["ZmFrZS1hdWRpbw=="])
    r = client.post("/api/v1/assistant/speak", json={"text": "hello there"}, headers=duty_h)
    assert r.status_code == 200
    assert r.json()["audios"] == ["ZmFrZS1hdWRpbw=="]


# ---------- grounding: the non-negotiable one ----------

def test_chat_without_selection_has_no_live_data(client, duty_h, monkeypatch):
    captured = {}

    def fake_chat(messages, model=sarvam.CHAT_MODEL, language="en"):
        captured["messages"] = messages
        return "Vishwas estimates how likely an issued forecast is to bust."
    monkeypatch.setattr(sarvam, "chat", fake_chat)

    r = client.post("/api/v1/assistant/chat", json={"text": "what is Vishwas"}, headers=duty_h)
    assert r.status_code == 200
    body = r.json()
    assert body["used_live_data"] is False
    assert body["grounding"] is None
    assert "NO LIVE DATA is attached for this question." in captured["messages"][0]["content"]
    assert "LIVE DATA IS ATTACHED for the user's current screen:" not in captured["messages"][0]["content"]


def test_chat_grounding_matches_real_service_data(client, duty_h, monkeypatch):
    """The non-negotiable check: a 'why' question's grounding must carry the SAME real
    number the dashboard would show for the same subdivision/lead_day/cycle -- fetched
    independently here, straight from VishwasService, not from the assistant's own output."""
    truth = service_registry.get_service().bust_probability("TN/PY", 4, CYCLE)
    expected_pct = round(truth["bust_probability"] * 100)
    expected_level = truth["level"]

    captured = {}

    def fake_chat(messages, model=sarvam.CHAT_MODEL, language="en"):
        captured["messages"] = messages
        return f"TN/PY is {expected_level} on Day 4 because the bust probability is {expected_pct}%."
    monkeypatch.setattr(sarvam, "chat", fake_chat)

    r = client.post("/api/v1/assistant/chat", json={
        "text": "why is TN/PY like this on day 4",
        "subdivision": "TN/PY", "lead_day": 4, "cycle": CYCLE,
    }, headers=duty_h)
    assert r.status_code == 200
    body = r.json()
    assert body["used_live_data"] is True
    grounding = body["grounding"]
    assert f"Bust probability: {expected_pct}%" in grounding
    assert f"Warning level: {expected_level}" in grounding
    # the same real number also has to be what was actually sent to the model, not just
    # what the route claims in its own response
    system_msg = captured["messages"][0]["content"]
    assert grounding in system_msg


def test_chat_bad_selection_falls_back_without_crashing(client, duty_h, monkeypatch):
    monkeypatch.setattr(sarvam, "chat", lambda messages, model=sarvam.CHAT_MODEL, language="en": "I don't have that selection.")
    r = client.post("/api/v1/assistant/chat", json={
        "text": "why is this like this",
        "subdivision": "NOT-A-CODE", "lead_day": 4, "cycle": CYCLE,
    }, headers=duty_h)
    assert r.status_code == 200
    assert r.json()["used_live_data"] is False


def test_chat_encoded_subdivision_code_edge_case(client, duty_h, monkeypatch):
    """J&K, same edge case as the ML routes -- grounding must resolve it correctly too."""
    truth = service_registry.get_service().bust_probability("J&K", 3, CYCLE)
    monkeypatch.setattr(sarvam, "chat", lambda messages, model=sarvam.CHAT_MODEL, language="en": "ok")
    r = client.post("/api/v1/assistant/chat", json={
        "text": "why", "subdivision": "J&K", "lead_day": 3, "cycle": CYCLE,
    }, headers=duty_h)
    assert r.status_code == 200
    body = r.json()
    assert body["used_live_data"] is True
    assert f"{round(truth['bust_probability'] * 100)}%" in body["grounding"]


def test_chat_site_facts_used_for_product_questions(client, duty_h, monkeypatch):
    captured = {}

    def fake_chat(messages, model=sarvam.CHAT_MODEL, language="en"):
        captured["system"] = messages[0]["content"]
        return "answer"
    monkeypatch.setattr(sarvam, "chat", fake_chat)
    r = client.post("/api/v1/assistant/chat", json={"text": "what does the big-miss watch mean"}, headers=duty_h)
    assert r.status_code == 200
    assert "BIG-MISS WATCH" in captured["system"]


def test_chat_reply_language_matches_ui_language(client, duty_h, monkeypatch):
    captured = {}

    def fake_chat(messages, model=sarvam.CHAT_MODEL, language="en"):
        captured["system"] = messages[0]["content"]
        return "ok"
    monkeypatch.setattr(sarvam, "chat", fake_chat)
    r = client.post("/api/v1/assistant/chat", json={"text": "what is a lead day", "ui_language": "hi"}, headers=duty_h)
    assert r.status_code == 200
    assert "Reply in Hindi" in captured["system"]
    assert "Devanagari" in captured["system"]
