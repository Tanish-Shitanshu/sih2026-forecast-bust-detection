from fastapi import APIRouter, Depends, HTTPException

import db
import auth
import jobs
import service_registry
from routers.settings import get_settings

router = APIRouter(tags=["retraining"], dependencies=[Depends(auth.require_role("admin"))])


@router.post("/api/v1/retraining", status_code=202)
def start_retraining(user=Depends(auth.require_role("admin"))):
    if jobs.is_running():
        raise HTTPException(409, "A retraining run is already in progress")

    conn = db.get_conn()
    approved = conn.execute("SELECT COUNT(*) AS n FROM outcomes WHERE status = 'approved'").fetchone()["n"]
    conn.close()
    db.audit_log(user.name, f"Started a retraining run ({approved} approved outcomes)")

    settings = get_settings()
    bust_overrides = {"min_error_mm": settings["error_threshold_mm"], "percentile": settings["regional_percentile"]}
    source = service_registry.current_source()
    try:
        job_id = jobs.start_job(source, approved, bust_overrides)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {"job_id": job_id, "status": "queued", "approved_outcomes_used": approved}


@router.get("/api/v1/retraining/{job_id}")
def retraining_status(job_id: str):
    row = jobs.get_job(job_id)
    if row is None:
        raise HTTPException(404, "No such retraining job")
    out = {"job_id": row["job_id"], "status": row["status"], "source": row["source"],
           "approved_outcomes_used": row["approved_outcomes_used"],
           "started_at": row["started_at"], "finished_at": row["finished_at"]}
    if row["metrics_json"]:
        import json
        out["metrics"] = json.loads(row["metrics_json"])
    if row["error"]:
        out["error"] = row["error"]
    return out
