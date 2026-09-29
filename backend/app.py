"""Vishwas backend. Run from inside backend/ (matches ml/serve.py's own
convention) so bare imports of the sibling modules (db, auth, jobs,
service_registry) resolve without extra packaging:

    cd backend
    ../ml/.venv/bin/python -m uvicorn app:app --port 8000

CORS is permissive (matches ml/serve.py's dev posture) -- fine for a
token-authenticated (not cookie-based) API with no credentialed requests.
"""
import os
import sys
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml"))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import db
import service_registry
from routers import auth_routes, ml_routes, outcomes, users, settings, retraining, audit


@asynccontextmanager
async def _lifespan(_: FastAPI):
    ml_subdivisions_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml", "data", "subdivisions.json")
    db.init_db(ml_subdivisions_path)
    service_registry.get_service()  # load the model once at startup, not on the first request
    yield


app = FastAPI(title="Vishwas API", version="1.0.0", lifespan=_lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(ValueError)
async def _bad_input(_: Request, e: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(e)})


@app.exception_handler(LookupError)
async def _not_found(_: Request, e: LookupError):
    return JSONResponse(status_code=404, content={"detail": str(e)})


@app.get("/health")
def health():
    svc = service_registry.get_service()
    return {
        "status": "ok",
        "source": service_registry.current_source(),
        "model_trained_at": svc.model.metadata.get("trained_at"),
        "cycles": len(svc.cycles()),
        "latest_cycle": f"{svc.cycles()[-1]:%Y-%m-%d}T00Z",
    }


app.include_router(auth_routes.router)
app.include_router(ml_routes.router)
app.include_router(outcomes.router)
app.include_router(users.router)
app.include_router(settings.router)
app.include_router(retraining.router)
app.include_router(audit.router)
