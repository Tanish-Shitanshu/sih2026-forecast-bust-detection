from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

import db
import auth

router = APIRouter(tags=["settings"], dependencies=[Depends(auth.require_role("admin"))])


@router.get("/api/v1/settings/bust-definition")
def read_bust_definition():
    """Not in the original 13-route contract, added so the admin settings panel can show the
    persisted values instead of guessing -- purely additive, same admin-only auth as the PUT."""
    return get_settings()


class SettingsIn(BaseModel):
    error_threshold_mm: int
    regional_percentile: int


@router.put("/api/v1/settings/bust-definition")
def set_bust_definition(body: SettingsIn, user=Depends(auth.require_role("admin"))):
    if not (1 <= body.error_threshold_mm <= 500):
        raise HTTPException(422, "error_threshold_mm must be between 1 and 500")
    if not (50 <= body.regional_percentile <= 99):
        raise HTTPException(422, "regional_percentile must be between 50 and 99")
    conn = db.get_conn()
    conn.execute(
        "INSERT INTO settings (id, error_threshold_mm, regional_percentile) VALUES (1, ?, ?) "
        "ON CONFLICT(id) DO UPDATE SET error_threshold_mm = excluded.error_threshold_mm, "
        "regional_percentile = excluded.regional_percentile",
        (body.error_threshold_mm, body.regional_percentile),
    )
    conn.commit()
    conn.close()
    db.audit_log(user.name, f"Saved the bust definition: error of at least {body.error_threshold_mm} mm "
                           f"or the regional {body.regional_percentile}th percentile")
    return {
        "saved": True,
        "error_threshold_mm": body.error_threshold_mm,
        "regional_percentile": body.regional_percentile,
        "applies_from": "next retraining run",
    }


def get_settings():
    conn = db.get_conn()
    row = conn.execute("SELECT error_threshold_mm, regional_percentile FROM settings WHERE id = 1").fetchone()
    conn.close()
    if row is None:
        return {"error_threshold_mm": 25, "regional_percentile": 95}
    return {"error_threshold_mm": row["error_threshold_mm"], "regional_percentile": row["regional_percentile"]}
