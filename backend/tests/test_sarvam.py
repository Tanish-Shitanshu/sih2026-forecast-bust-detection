"""Unit tests for sarvam.py's own logic (content-type normalization, TTS chunking) --
independent of the assistant routes, and without hitting the network (httpx.post is
monkeypatched)."""
import sarvam


class FakeResponse:
    def __init__(self, status_code=200, json_body=None, text=""):
        self.status_code = status_code
        self._json = json_body or {}
        self.text = text

    def json(self):
        return self._json


def test_transcribe_strips_codec_parameter_from_content_type(monkeypatch):
    """Sarvam rejects "audio/webm;codecs=opus" (what Chrome's MediaRecorder actually sends)
    but accepts the bare "audio/webm" -- this was a real bug found via live testing."""
    captured = {}

    def fake_post(url, timeout, headers, data, files):
        captured["files"] = files
        return FakeResponse(200, {"transcript": "hello", "language_code": "en-IN", "language_probability": 0.9})

    monkeypatch.setattr(sarvam.httpx, "post", fake_post)
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    sarvam.transcribe(b"fake-audio", "q.webm", "audio/webm;codecs=opus")
    sent_content_type = captured["files"]["file"][2]
    assert sent_content_type == "audio/webm"


def test_transcribe_leaves_bare_content_type_unchanged(monkeypatch):
    captured = {}

    def fake_post(url, timeout, headers, data, files):
        captured["files"] = files
        return FakeResponse(200, {"transcript": "hello"})

    monkeypatch.setattr(sarvam.httpx, "post", fake_post)
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    sarvam.transcribe(b"fake-audio", "q.wav", "audio/wav")
    assert captured["files"]["file"][2] == "audio/wav"


def test_chat_raises_on_empty_content_with_no_length_reason(monkeypatch):
    """Empty content with a normal (non-"length") finish_reason is a genuinely unexpected
    shape -- still a hard error, distinct from the reasoning-exhaustion case below."""
    def fake_post(url, timeout, headers, json):
        return FakeResponse(200, {"choices": [{"finish_reason": "stop", "message": {"content": ""}}]})

    monkeypatch.setattr(sarvam.httpx, "post", fake_post)
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    try:
        sarvam.chat([{"role": "user", "content": "hi"}])
        assert False, "expected SarvamError"
    except sarvam.SarvamError as e:
        assert "unexpected Sarvam chat response shape" in str(e)


def test_chat_raises_on_null_content_with_no_length_reason(monkeypatch):
    def fake_post(url, timeout, headers, json):
        return FakeResponse(200, {"choices": [{"finish_reason": "stop", "message": {"content": None}}]})

    monkeypatch.setattr(sarvam.httpx, "post", fake_post)
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    try:
        sarvam.chat([{"role": "user", "content": "hi"}])
        assert False, "expected SarvamError"
    except sarvam.SarvamError as e:
        assert "unexpected Sarvam chat response shape" in str(e)


def test_chat_reasoning_model_exhausts_budget_then_succeeds_on_retry(monkeypatch):
    """The exact real bug: sarvam-105b spends its whole token budget on internal
    reasoning_content and stops at finish_reason "length" with content: None. The first
    attempt hits this; the second (more headroom, lower effort) succeeds -- confirms the
    retry actually recovers a real answer, not just handles the failure."""
    seen_attempts = []

    def fake_post(url, timeout, headers, json):
        seen_attempts.append((json["max_tokens"], json["reasoning_effort"]))
        if len(seen_attempts) == 1:
            return FakeResponse(200, {"choices": [{
                "finish_reason": "length",
                "message": {"content": None, "reasoning_content": "Let me think about this platform..." * 50},
            }]})
        return FakeResponse(200, {"choices": [{
            "finish_reason": "stop",
            "message": {"content": "Vishwas is a Forecast Trust Console for NCMRWF."},
        }]})

    monkeypatch.setattr(sarvam.httpx, "post", fake_post)
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    answer = sarvam.chat([{"role": "user", "content": "मुझे इस प्लेटफॉर्म के बारे में बताओ"}])
    assert answer == "Vishwas is a Forecast Trust Console for NCMRWF."
    assert len(seen_attempts) == 2
    assert seen_attempts[0] == (4096, "low")
    assert seen_attempts[1][0] > seen_attempts[0][0]  # strictly more headroom on retry


def test_chat_reasoning_model_exhausts_every_attempt_returns_clean_fallback(monkeypatch):
    """Both attempts hit finish_reason "length" with no content -- the caller must get a
    short, plain, translated apology, never the raw payload or the reasoning trace."""
    def fake_post(url, timeout, headers, json):
        return FakeResponse(200, {"choices": [{
            "finish_reason": "length",
            "message": {"content": None, "reasoning_content": "internal chain of thought, never shown to users"},
        }]})

    monkeypatch.setattr(sarvam.httpx, "post", fake_post)
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")

    answer_en = sarvam.chat([{"role": "user", "content": "what is Vishwas"}], language="en")
    assert answer_en == "Could not get an answer just now. Please try again."
    assert "reasoning" not in answer_en.lower()
    assert "internal chain of thought" not in answer_en

    answer_hi = sarvam.chat([{"role": "user", "content": "मुझे इस प्लेटफॉर्म के बारे में बताओ"}], language="hi")
    assert answer_hi == "अभी उत्तर नहीं मिल सका। कृपया फिर से प्रयास करें।"
    assert "internal chain of thought" not in answer_hi


def test_speak_chunks_long_text_under_the_2500_char_limit(monkeypatch):
    calls = []

    def fake_post(url, timeout, headers, json):
        calls.append(json["text"])
        return FakeResponse(200, {"audios": ["YWJj"]})

    monkeypatch.setattr(sarvam.httpx, "post", fake_post)
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    long_text = ("This is a sentence. " * 200).strip()  # well over 2500 chars
    assert len(long_text) > 2500
    audios = sarvam.speak(long_text)
    assert len(calls) > 1
    assert all(len(c) <= 2500 for c in calls)
    assert len(audios) == len(calls)


def test_speak_short_text_is_a_single_call(monkeypatch):
    calls = []

    def fake_post(url, timeout, headers, json):
        calls.append(json["text"])
        return FakeResponse(200, {"audios": ["YWJj"]})

    monkeypatch.setattr(sarvam.httpx, "post", fake_post)
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    sarvam.speak("A short answer.")
    assert len(calls) == 1
    assert calls[0] == "A short answer."


def test_missing_key_raises_before_any_network_call(monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    called = {"n": 0}

    def fake_post(*a, **k):
        called["n"] += 1
        return FakeResponse(200, {})

    monkeypatch.setattr(sarvam.httpx, "post", fake_post)
    try:
        sarvam.chat([{"role": "user", "content": "hi"}])
        assert False, "expected SarvamError"
    except sarvam.SarvamError as e:
        assert "SARVAM_API_KEY" in str(e)
    assert called["n"] == 0
