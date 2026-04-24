"""Minimal JWT + bcrypt auth for the FYP demo.

Stateless: tokens carry `account_id` and `role`. No refresh / revocation.
Reads DATABASE_URL via AIConfig so it shares the Supabase connection.
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass

import psycopg2
from argon2 import PasswordHasher, Type
from argon2.exceptions import VerifyMismatchError
from fastapi import Depends, Header, HTTPException
from jose import JWTError, jwt

SECRET_KEY = os.getenv("JWT_SECRET", "fyp-demo-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_TTL_HOURS = 12

# Argon2id — OWASP-recommended defaults (64 MiB, t=3, p=4).
_ph = PasswordHasher(type=Type.ID, memory_cost=65536, time_cost=3, parallelism=4)


@dataclass
class CurrentUser:
    account_id: int
    role: str
    email: str
    full_name: str | None = None


# ── password helpers ───────────────────────────────────────────────
def hash_password(raw: str) -> str:
    return _ph.hash(raw)


def verify_password(raw: str, hashed: str) -> bool:
    try:
        _ph.verify(hashed, raw)
        return True
    except (VerifyMismatchError, Exception):
        return False


def needs_rehash(hashed: str) -> bool:
    """True if the stored hash uses outdated Argon2 parameters."""
    try:
        return _ph.check_needs_rehash(hashed)
    except Exception:
        return True


# ── token helpers ──────────────────────────────────────────────────
def create_token(account_id: int, role: str, email: str) -> str:
    payload = {
        "sub": str(account_id),
        "role": role,
        "email": email,
        "exp": dt.datetime.utcnow() + dt.timedelta(hours=ACCESS_TOKEN_TTL_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def _decode(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError as exc:
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}")


# ── FastAPI dependency ─────────────────────────────────────────────
def get_current_user(authorization: str | None = Header(default=None)) -> CurrentUser:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    data = _decode(token)
    return CurrentUser(
        account_id=int(data["sub"]),
        role=data.get("role", "student"),
        email=data.get("email", ""),
    )


def require_role(*roles: str):
    def _dep(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if user.role not in roles:
            raise HTTPException(status_code=403, detail="Forbidden")
        return user
    return _dep


# ── DB auth lookup (synchronous; shares DATABASE_URL) ──────────────
def authenticate(database_url: str, email: str, password: str) -> CurrentUser | None:
    sql = """
        SELECT ua.accountid, up.role, ua.email, ua.password_hash, pi.full_name
        FROM user_account ua
        JOIN user_profiles up ON up.profileid = ua.profileid
        LEFT JOIN personal_info pi ON pi.accountid = ua.accountid
        WHERE lower(ua.email) = lower(%s)
        LIMIT 1
    """
    with psycopg2.connect(database_url) as conn, conn.cursor() as cur:
        cur.execute(sql, (email,))
        row = cur.fetchone()
    if not row:
        return None
    account_id, role, em, pw_hash, full_name = row
    if not verify_password(password, pw_hash):
        return None

    # Transparently upgrade legacy / weaker Argon2 hashes on successful login.
    if needs_rehash(pw_hash):
        try:
            with psycopg2.connect(database_url) as conn, conn.cursor() as cur:
                cur.execute(
                    "UPDATE user_account SET password_hash = %s WHERE accountid = %s",
                    (hash_password(password), account_id),
                )
        except Exception:
            pass  # don't block login on a rehash failure

    return CurrentUser(account_id=account_id, role=role, email=em, full_name=full_name)
