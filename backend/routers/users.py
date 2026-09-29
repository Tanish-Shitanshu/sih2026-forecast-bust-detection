import re

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

import db
import auth

router = APIRouter(tags=["users"], dependencies=[Depends(auth.require_role("admin"))])

USER_ID_RE = re.compile(r"^[a-z][a-z0-9]{2,19}$")


@router.get("/api/v1/users")
def list_users():
    conn = db.get_conn()
    rows = conn.execute("SELECT id, name, role, active FROM users ORDER BY id").fetchall()
    conn.close()
    items = [{"id": r["id"], "name": r["name"], "role": r["role"], "active": bool(r["active"])} for r in rows]
    return {"count": len(items), "items": items}


class UserIn(BaseModel):
    id: str
    name: str
    role: str


@router.post("/api/v1/users", status_code=201)
def create_user(body: UserIn, user=Depends(auth.require_role("admin"))):
    if not USER_ID_RE.match(body.id):
        raise HTTPException(422, "The user ID must be 3 to 20 lowercase letters or digits, starting with a letter.")
    if body.role not in auth.ROLE_KEYS:
        raise HTTPException(422, "role must be duty, senior, admin or observer")
    conn = db.get_conn()
    exists = conn.execute("SELECT 1 FROM users WHERE id = ?", (body.id,)).fetchone()
    if exists:
        conn.close()
        raise HTTPException(409, "That user ID already exists")
    conn.execute(
        "INSERT INTO users (id, name, role, password_hash, active) VALUES (?, ?, ?, ?, 1)",
        (body.id, body.name, body.role, db.hash_password(db.DEMO_PASSWORD)),
    )
    conn.commit()
    conn.close()
    db.audit_log(user.name, f"Added user {body.name} ({body.id}) as {body.role}")
    return {"id": body.id, "name": body.name, "role": body.role, "active": True}


class UserPatch(BaseModel):
    role: str | None = None
    active: bool | None = None


@router.patch("/api/v1/users/{user_id}")
def patch_user(user_id: str, body: UserPatch, user=Depends(auth.require_role("admin"))):
    if user_id == user.id:
        raise HTTPException(403, "Administrators cannot change or deactivate their own account")
    if body.role is not None and body.role not in auth.ROLE_KEYS:
        raise HTTPException(422, "role must be duty, senior, admin or observer")
    conn = db.get_conn()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if row is None:
        conn.close()
        raise HTTPException(404, "No such user")
    new_role = body.role if body.role is not None else row["role"]
    new_active = int(body.active) if body.active is not None else row["active"]
    conn.execute("UPDATE users SET role = ?, active = ? WHERE id = ?", (new_role, new_active, user_id))
    conn.commit()
    conn.close()
    if body.role is not None and body.role != row["role"]:
        db.audit_log(user.name, f"Changed the role of {row['name']} from {row['role']} to {new_role}")
    if body.active is not None and bool(body.active) != bool(row["active"]):
        db.audit_log(user.name, f"{'Reactivated' if body.active else 'Deactivated'} the account of {row['name']}")
    return {"id": user_id, "role": new_role, "active": bool(new_active)}
