"""Development API server for the ML endpoints, on the frontend API page's paths.

    cd ml && .venv/Scripts/python -m uvicorn serve:app --port 8000
    VISHWAS_SOURCE=ncmrwf .venv/Scripts/python -m uvicorn serve:app --port 8000   # pick the model

Open http://localhost:8000/docs for the interactive list. This is a thin wrapper over
vishwas_ml.service.VishwasService so the backend team can see every response live; real
auth, the users/audit/settings endpoints and persistent outcome storage belong in backend/.
Role checks here read an optional `X-Role` header (duty, senior, admin, observer) in place
of the sign-in token.

Subdivision codes contain "/" and "&" (e.g. TN/PY, J&K): URL-encode them (TN%2FPY, J%26K).
"""
import os
from functools import lru_cache

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from vishwas_ml.config import load_config
from vishwas_ml.feedback import read_outcomes
from vishwas_ml.service import VishwasService

app = FastAPI(title="Vishwas ML API (dev)", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
ROLES = {"duty", "senior", "admin", "observer"}


@lru_cache(maxsize=1)
def svc():
    return VishwasService(cfg=load_config(source=os.environ.get("VISHWAS_SOURCE")))


@app.exception_handler(ValueError)
async def _bad_input(_: Request, e: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(e)})


@app.exception_handler(LookupError)
async def _not_found(_: Request, e: LookupError):
    return JSONResponse(status_code=404, content={"detail": str(e)})


def _role(x_role, allowed=ROLES):
    role = (x_role or "observer").lower()
    if role not in ROLES:
        raise HTTPException(401, "unknown role")
    if role not in allowed:
        raise HTTPException(403, "The role cannot perform this action")
    return role


@app.get("/health")
def health():
    s = svc()
    return {"status": "ok", "source": s.cfg.get("source"), "model_trained_at": s.model.metadata.get("trained_at"),
            "cycles": len(s.cycles()), "latest_cycle": f"{s.cycles()[-1]:%Y-%m-%d}T00Z"}


@app.get("/api/v1/cycles")
def cycles():
    """Available forecast issue dates (the demo replays historical cycles)."""
    return {"items": [f"{c:%Y-%m-%d}T00Z" for c in svc().cycles()]}


@app.get("/api/v1/confidence-map")
def confidence_map(lead_day: int = Query(..., ge=1, le=10), cycle: str | None = None):
    return svc().confidence_map(lead_day, cycle)


@app.get("/api/v1/bust-probability")
def bust_probability(subdivision: str, lead_day: int = Query(..., ge=1, le=10), cycle: str | None = None):
    return svc().bust_probability(subdivision, lead_day, cycle)


@app.get("/api/v1/subdivisions/{code:path}/explain")
def explain(code: str, lead_day: int = Query(..., ge=1, le=10), cycle: str | None = None):
    return svc().explain(code, lead_day, cycle)


@app.get("/api/v1/model-trust")
def model_trust(lead_day: int | None = Query(None, ge=1, le=10), subdivision: str | None = None,
                cycle: str | None = None):
    if lead_day is None and subdivision is None:
        raise HTTPException(422, "give lead_day or subdivision")
    return svc().model_trust(lead_day=lead_day, subdivision=subdivision, cycle=cycle)


@app.get("/api/v1/analogs")
def analogs(subdivision: str, lead_day: int = Query(..., ge=1, le=10), cycle: str | None = None,
            k: int = Query(5, ge=1, le=25)):
    return svc().analogs(subdivision, lead_day, cycle, k=k)


@app.get("/api/v1/action")
def action(subdivision: str, lead_day: int = Query(..., ge=1, le=10), cycle: str | None = None):
    return svc().action(subdivision, lead_day, cycle)


class OutcomeIn(BaseModel):
    subdivision: str
    lead_day: int = Field(ge=1, le=10)
    outcome: str
    note: str | None = Field(None, max_length=80)
    cycle: str | None = None


@app.post("/api/v1/outcomes", status_code=201)
def record_outcome(body: OutcomeIn, x_role: str | None = Header(None), x_user: str | None = Header(None)):
    _role(x_role, {"duty", "senior"})
    return svc().record_outcome(body.subdivision, body.lead_day, body.outcome, note=body.note,
                                submitted_by=x_user or "forecaster", cycle=body.cycle)


@app.get("/api/v1/outcomes")
def list_outcomes(status: str | None = None, subdivision: str | None = None):
    d = read_outcomes(svc().feedback_path)
    if status:
        d = d[d["status"] == status]
    if subdivision:
        d = d[d["subdivision"] == subdivision]
    cols = ["id", "subdivision", "lead_day", "predicted_bust_probability", "outcome", "submitted_by", "status"]
    items = d[[c for c in cols if c in d]].to_dict(orient="records")
    return {"count": len(items), "items": items}
