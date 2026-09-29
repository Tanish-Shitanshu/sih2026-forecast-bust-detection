import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml"))
from vishwas_ml.feedback import validate_outcome  # noqa: E402

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

import db
import auth
import service_registry

router = APIRouter(tags=["outcomes"])


class OutcomeIn(BaseModel):
    subdivision: str
    lead_day: int = Field(ge=1, le=10)
    outcome: str
    note: str | None = Field(None, max_length=80)
    cycle: str | None = None


@router.post("/api/v1/outcomes", status_code=201)
def record_outcome(body: OutcomeIn, user=Depends(auth.require_role("duty", "senior"))):
    validate_outcome(body.model_dump())  # raises ValueError -> 422, via the global handler
    # svc.bust_probability both resolves/validates the subdivision (ValueError -> 422 on
    # an unknown code) and gives the real predicted probability to store against this entry.
    pred = service_registry.get_service().bust_probability(body.subdivision, body.lead_day, body.cycle)

    conn = db.get_conn()
    cur = conn.execute(
        "INSERT INTO outcomes (date, subdivision, subdivision_name, lead_day, predicted_bust_probability, "
        "outcome, note, submitted_by, status, cycle) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)",
        (db.now_iso(), pred["subdivision"], None, body.lead_day, pred["bust_probability"],
         body.outcome, body.note, user.id, body.cycle),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return {
        "id": row_id,
        "subdivision": pred["subdivision"],
        "lead_day": body.lead_day,
        "predicted_bust_probability": pred["bust_probability"],
        "outcome": body.outcome,
        "status": "pending",
        "submitted_by": user.id,
    }


@router.get("/api/v1/outcomes")
def list_outcomes(status: str | None = None, subdivision: str | None = None,
                   user=Depends(auth.require_any_role)):
    conn = db.get_conn()
    # LEFT JOINs resolve submitted_by/reviewed_by ids to display names for the feedback log --
    # a purely additive field on this backend-owned route, open to all 4 roles like the rest of
    # this endpoint already is (GET /api/v1/users itself stays admin-only per the API contract).
    q = ("SELECT o.id, o.date, o.subdivision, o.lead_day, o.predicted_bust_probability, o.outcome, "
         "o.note, o.submitted_by, su.name AS submitted_by_name, o.status, o.reviewed_by, "
         "ru.name AS reviewed_by_name, o.cycle "
         "FROM outcomes o LEFT JOIN users su ON su.id = o.submitted_by "
         "LEFT JOIN users ru ON ru.id = o.reviewed_by WHERE 1=1")
    params = []
    if status:
        q += " AND o.status = ?"
        params.append(status)
    if subdivision:
        q += " AND o.subdivision = ?"
        params.append(subdivision)
    q += " ORDER BY o.id DESC"
    rows = conn.execute(q, params).fetchall()
    conn.close()
    items = [dict(r) for r in rows]
    return {"count": len(items), "items": items}


class ReviewIn(BaseModel):
    decision: str  # "approve" | "reject"


@router.post("/api/v1/outcomes/{outcome_id}/review")
def review_outcome(outcome_id: int, body: ReviewIn, user=Depends(auth.require_role("senior"))):
    if body.decision not in ("approve", "reject"):
        raise HTTPException(422, "decision must be approve or reject")
    conn = db.get_conn()
    row = conn.execute("SELECT * FROM outcomes WHERE id = ?", (outcome_id,)).fetchone()
    if row is None:
        conn.close()
        raise HTTPException(404, "No such outcome")
    if row["status"] != "pending":
        conn.close()
        raise HTTPException(409, "The entry is not pending")
    if row["submitted_by"] == user.id:
        conn.close()
        raise HTTPException(403, "Only senior forecasters may review, and not their own entries")
    new_status = "approved" if body.decision == "approve" else "rejected"
    conn.execute("UPDATE outcomes SET status = ?, reviewed_by = ? WHERE id = ?", (new_status, user.id, outcome_id))
    conn.commit()
    conn.close()
    return {"id": outcome_id, "status": new_status, "reviewed_by": user.id}
