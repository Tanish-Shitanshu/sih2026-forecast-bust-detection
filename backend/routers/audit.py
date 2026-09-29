from fastapi import APIRouter, Depends, Query

import db
import auth

router = APIRouter(tags=["audit"], dependencies=[Depends(auth.require_role("admin"))])


@router.get("/api/v1/audit")
def list_audit(limit: int = Query(50, ge=1, le=50)):
    conn = db.get_conn()
    rows = conn.execute("SELECT time, actor, action FROM audit ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return {"items": [{"time": r["time"], "actor": r["actor"], "action": r["action"]} for r in rows]}
