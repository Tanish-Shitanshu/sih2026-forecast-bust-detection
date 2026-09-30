"""Thin client for the Sarvam AI APIs: chat completions, speech-to-text, text-to-speech.
Every call goes through the backend only -- SARVAM_API_KEY never reaches the browser, and
callers (routers/assistant.py) turn SarvamError into a clean 502, never a raw crash.
"""
import os
import re

import httpx

BASE = "https://api.sarvam.ai"
CHAT_URL = f"{BASE}/v1/chat/completions"
STT_URL = f"{BASE}/speech-to-text"
TTS_URL = f"{BASE}/text-to-speech"
TIMEOUT = 25.0
CHAT_TIMEOUT = 55.0  # sarvam-105b with a long grounded system prompt is measurably slower
CHAT_MODEL = "sarvam-105b"
STT_MODEL = "saaras:v4"


class SarvamError(Exception):
    """Network failure, timeout, non-2xx response, or an unexpected response shape."""


def _key() -> str:
    key = os.environ.get("SARVAM_API_KEY")
    if not key:
        raise SarvamError("SARVAM_API_KEY is not set on the server")
    return key


# sarvam-105b is a reasoning model: it can spend its entire token budget on an internal
# reasoning_content chain and stop at finish_reason "length" having never emitted an actual
# answer (content: None). Reproduced live with the Hindi voice question "मुझे इस प्लेटफॉर्म
# के बारे में बताओ". These answers are meant to be short (2-4 sentences per the system
# prompt), so reasoning_effort is kept low and max_tokens generous; CHAT_ATTEMPTS is tried
# in order, each attempt only kept if the previous one ran out of budget mid-reasoning.
CHAT_ATTEMPTS = [
    {"max_tokens": 4096, "reasoning_effort": "low"},
    {"max_tokens": 6144, "reasoning_effort": "minimal"},
]

_CHAT_FALLBACK = {
    "en": "Could not get an answer just now. Please try again.",
    "hi": "अभी उत्तर नहीं मिल सका। कृपया फिर से प्रयास करें।",
}


def _chat_once(messages: list[dict], model: str, max_tokens: int, reasoning_effort: str) -> dict:
    try:
        r = httpx.post(CHAT_URL, timeout=CHAT_TIMEOUT,
                        headers={"api-subscription-key": _key(), "Content-Type": "application/json"},
                        json={"model": model, "messages": messages, "max_tokens": max_tokens,
                              "reasoning_effort": reasoning_effort})
    except httpx.HTTPError as e:
        raise SarvamError(f"could not reach Sarvam chat completions: {e}") from e
    if r.status_code >= 400:
        raise SarvamError(f"Sarvam chat completions returned {r.status_code}: {r.text[:300]}")
    return r.json()


def chat(messages: list[dict], model: str = CHAT_MODEL, language: str = "en") -> str:
    """language: short UI code ("en"/"hi"), used only to pick the fallback message if every
    attempt exhausts its budget on reasoning without ever emitting content -- a known,
    explainable failure, so it returns a clean apology rather than raising (never the raw
    API payload or reasoning trace to the caller). A genuine network/HTTP failure from
    _chat_once still propagates immediately as a SarvamError, unchanged."""
    for i, attempt in enumerate(CHAT_ATTEMPTS):
        body = _chat_once(messages, model, attempt["max_tokens"], attempt["reasoning_effort"])
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content")
        finish_reason = choice.get("finish_reason")
        if content and content.strip():
            return content
        if finish_reason == "length":
            if i + 1 < len(CHAT_ATTEMPTS):
                continue  # spent the whole budget reasoning -- retry with more headroom
            return _CHAT_FALLBACK.get(language, _CHAT_FALLBACK["en"])
        raise SarvamError(f"unexpected Sarvam chat response shape: {body}")
    return _CHAT_FALLBACK.get(language, _CHAT_FALLBACK["en"])  # unreachable; satisfies type checkers


def transcribe(audio_bytes: bytes, filename: str, content_type: str, language_code: str | None = None) -> dict:
    """language_code: hi-IN, en-IN, ... or None/omitted to auto-detect ('unknown').

    Browsers' MediaRecorder reports a codec-qualified type (e.g.
    "audio/webm;codecs=opus"); Sarvam's file-type check only accepts the bare
    "audio/webm", so the codec parameter is stripped before upload."""
    bare_type = content_type.split(";")[0].strip() if content_type else content_type
    try:
        r = httpx.post(STT_URL, timeout=TIMEOUT,
                        headers={"api-subscription-key": _key()},
                        data={"model": STT_MODEL, "language_code": language_code or "unknown"},
                        files={"file": (filename, audio_bytes, bare_type)})
    except httpx.HTTPError as e:
        raise SarvamError(f"could not reach Sarvam speech-to-text: {e}") from e
    if r.status_code >= 400:
        raise SarvamError(f"Sarvam speech-to-text returned {r.status_code}: {r.text[:300]}")
    body = r.json()
    return {
        "transcript": body.get("transcript", ""),
        "language_code": body.get("language_code"),
        "language_probability": body.get("language_probability"),
    }


_MAX_TTS_CHARS = 2500


def _chunks(text: str, limit: int = _MAX_TTS_CHARS) -> list[str]:
    """Splits on sentence boundaries (Latin + Devanagari) so each chunk stays under Sarvam's
    2500-character-per-call limit. Assistant answers are short, so this is almost always a
    single chunk -- it exists for the rare long answer, not the common case."""
    if len(text) <= limit:
        return [text]
    sentences = re.split(r"(?<=[.!?।])\s+", text)
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) + 1 > limit:
            if cur:
                chunks.append(cur.strip())
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(cur.strip())
    return chunks or [text[:limit]]


def speak(text: str, language_code: str = "en-IN", speaker: str | None = None) -> list[str]:
    """Returns base64 audio clips, one per <=2500-char chunk, in reading order."""
    audios = []
    for chunk in _chunks(text):
        payload = {"text": chunk, "language_code": language_code}
        if speaker:
            payload["speaker"] = speaker
        try:
            r = httpx.post(TTS_URL, timeout=TIMEOUT,
                            headers={"api-subscription-key": _key(), "Content-Type": "application/json"},
                            json=payload)
        except httpx.HTTPError as e:
            raise SarvamError(f"could not reach Sarvam text-to-speech: {e}") from e
        if r.status_code >= 400:
            raise SarvamError(f"Sarvam text-to-speech returned {r.status_code}: {r.text[:300]}")
        body = r.json()
        clips = body.get("audios") or []
        if not clips:
            raise SarvamError(f"unexpected Sarvam text-to-speech response shape: {body}")
        audios.extend(clips)
    return audios
