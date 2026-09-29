from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import db
import auth

router = APIRouter(tags=["auth"])


class LoginIn(BaseModel):
    id: str
    password: str


@router.post("/api/v1/auth/login")
def login(body: LoginIn):
    conn = db.get_conn()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (body.id.strip().lower(),)).fetchone()
    conn.close()
    if row is None or not db.verify_password(body.password, row["password_hash"]):
        raise HTTPException(401, "User ID or password is incorrect.")
    if not row["active"]:
        raise HTTPException(401, "This account is deactivated. Ask an administrator to reactivate it.")
    token = auth.issue_token(row["id"], row["role"])
    return {"token": token, "id": row["id"], "name": row["name"], "role": row["role"]}
