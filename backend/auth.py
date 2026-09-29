"""JWT issuance/verification and role-check dependencies.

The frontend's API page assumes "requests carry a token issued at sign-in,
and the role in the token decides access" but doesn't list a login route
(the current frontend validates client-side against a hardcoded password).
POST /api/v1/auth/login below is the minimal addition needed to make that
assumption real -- see backend/README.md for why.
"""
import os
import datetime

import jwt
from fastapi import Depends, Header, HTTPException

import db

SECRET = os.environ.get("VISHWAS_JWT_SECRET", "vishwas-hackathon-demo-secret-change-in-production")
ALGORITHM = "HS256"
TOKEN_TTL_HOURS = 12

ROLE_KEYS = {"duty", "senior", "admin", "observer"}


def issue_token(user_id: str, role: str) -> str:
    payload = {
        "sub": user_id,
        "role": role,
        "exp": datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=TOKEN_TTL_HOURS),
        "iat": datetime.datetime.now(datetime.UTC),
    }
    return jwt.encode(payload, SECRET, algorithm=ALGORITHM)


def _decode(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired, sign in again")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token")


class CurrentUser:
    def __init__(self, id: str, role: str, name: str):
        self.id = id
        self.role = role
        self.name = name


def get_current_user(authorization: str | None = Header(None)) -> CurrentUser:
    """Bearer token -> CurrentUser. Also re-checks the account is still
    active on every request (a deactivated account's old tokens stop
    working immediately, not just at next login)."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing bearer token")
    token = authorization.split(" ", 1)[1]
    payload = _decode(token)
    conn = db.get_conn()
    row = conn.execute("SELECT name, active FROM users WHERE id = ?", (payload["sub"],)).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(401, "Unknown user")
    if not row["active"]:
        raise HTTPException(401, "This account is deactivated")
    return CurrentUser(id=payload["sub"], role=payload["role"], name=row["name"])


def require_role(*allowed: str):
    """FastAPI dependency factory: 403 if the caller's role isn't in `allowed`."""
    allowed_set = set(allowed)

    def _check(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if user.role not in allowed_set:
            raise HTTPException(403, "The role cannot perform this action")
        return user

    return _check


def require_any_role(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    """Any authenticated, active user (all 4 roles) -- used for the 'all' routes."""
    return user
