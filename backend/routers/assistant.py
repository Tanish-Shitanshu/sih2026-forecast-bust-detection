"""Sarvam-backed assistant: product help (grounded on real site content) and "why is this
like this" explanations (grounded on real VishwasService data for the caller's current
dashboard selection). All three Sarvam calls happen server-side only -- SARVAM_API_KEY
never reaches the browser -- and any Sarvam failure becomes a clean 502, never a crash.
Open to all 4 roles, same as the ML routes.
"""
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

import auth
import sarvam
import service_registry
from assistant_content import SITE_FACTS, SYSTEM_PROMPT, live_data_block

router = APIRouter(tags=["assistant"], dependencies=[Depends(auth.require_any_role)])

_LANGUAGE_NAMES = {"hi": "Hindi, written in Devanagari script (not Romanized/Hinglish)", "en": "English"}


@router.post("/api/v1/assistant/transcribe")
async def transcribe(file: UploadFile = File(...), language_code: str | None = Form(None)):
    audio = await file.read()
    if not audio:
        raise HTTPException(422, "No audio received")
    try:
        return sarvam.transcribe(audio, file.filename or "audio.webm",
                                  file.content_type or "audio/webm", language_code)
    except sarvam.SarvamError as e:
        raise HTTPException(502, str(e))


class ChatIn(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    cycle: str | None = None
    subdivision: str | None = None
    lead_day: int | None = Field(None, ge=1, le=10)
    ui_language: str | None = None  # "en" or "hi", matches the site's own toggle


@router.post("/api/v1/assistant/chat")
def chat(body: ChatIn):
    grounding = None
    used_live_data = False
    if body.subdivision and body.lead_day:
        try:
            svc = service_registry.get_service()
            bp = svc.bust_probability(body.subdivision, body.lead_day, body.cycle)
            ex = svc.explain(body.subdivision, body.lead_day, body.cycle)
            grounding = live_data_block(bp, ex, body.subdivision)
            used_live_data = True
        except (ValueError, LookupError):
            # Bad/stale selection (e.g. an unknown cycle) -- fall through with no LIVE DATA
            # block; the system prompt tells the model to say so rather than invent a number.
            grounding = None

    language = _LANGUAGE_NAMES.get(body.ui_language or "en", "English")
    live_data_status = f"LIVE DATA IS ATTACHED for the user's current screen:\n{grounding}" if grounding \
        else "NO LIVE DATA is attached for this question."
    system = SYSTEM_PROMPT.format(site_facts=SITE_FACTS, language=language, live_data_status=live_data_status)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": body.text}]
    try:
        answer = sarvam.chat(messages, language=body.ui_language or "en")
    except sarvam.SarvamError as e:
        raise HTTPException(502, str(e))
    return {"answer": answer, "used_live_data": used_live_data, "grounding": grounding}


class SpeakIn(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    language_code: str = "en-IN"


@router.post("/api/v1/assistant/speak")
def speak(body: SpeakIn):
    try:
        audios = sarvam.speak(body.text, body.language_code)
    except sarvam.SarvamError as e:
        raise HTTPException(502, str(e))
    return {"audios": audios}
