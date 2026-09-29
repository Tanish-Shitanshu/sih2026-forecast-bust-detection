"""Thin wrappers over VishwasService -- every response here is the service's
dict passed through unchanged, per the ML handoff ("Pass it through
unchanged"). All require sign-in (any of the 4 roles), matching the API
page's roles:ROLE_KEYS for these routes.
"""
from fastapi import APIRouter, Depends, HTTPException, Query

import auth
import service_registry

router = APIRouter(tags=["ml"], dependencies=[Depends(auth.require_any_role)])


@router.get("/api/v1/cycles")
def cycles():
    return {"items": [f"{c:%Y-%m-%d}T00Z" for c in service_registry.get_service().cycles()]}


@router.get("/api/v1/confidence-map")
def confidence_map(lead_day: int = Query(..., ge=1, le=10), cycle: str | None = None):
    return service_registry.get_service().confidence_map(lead_day, cycle)


@router.get("/api/v1/bust-probability")
def bust_probability(subdivision: str, lead_day: int = Query(..., ge=1, le=10), cycle: str | None = None):
    return service_registry.get_service().bust_probability(subdivision, lead_day, cycle)


@router.get("/api/v1/subdivisions/{code:path}/explain")
def explain(code: str, lead_day: int = Query(..., ge=1, le=10), cycle: str | None = None):
    return service_registry.get_service().explain(code, lead_day, cycle)


@router.get("/api/v1/model-trust")
def model_trust(lead_day: int | None = Query(None, ge=1, le=10), subdivision: str | None = None,
                 cycle: str | None = None):
    if lead_day is None and subdivision is None:
        raise HTTPException(422, "give lead_day or subdivision")
    return service_registry.get_service().model_trust(lead_day=lead_day, subdivision=subdivision, cycle=cycle)


@router.get("/api/v1/analogs")
def analogs(subdivision: str, lead_day: int = Query(..., ge=1, le=10), cycle: str | None = None,
            k: int = Query(5, ge=1, le=25)):
    return service_registry.get_service().analogs(subdivision, lead_day, cycle, k=k)


@router.get("/api/v1/action")
def action(subdivision: str, lead_day: int = Query(..., ge=1, le=10), cycle: str | None = None):
    return service_registry.get_service().action(subdivision, lead_day, cycle)
