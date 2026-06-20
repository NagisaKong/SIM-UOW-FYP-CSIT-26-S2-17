"""Class 1: userInformation

Covers UC:  U01 Student Login, U02 Student Logout(boundary),
            U10 Teacher Login, U11 Teacher Logout(boundary),
            U17 Admin Login, U18 Admin Logout(boundary),
            U20 Manage User Accounts.

Absorbs former api/auth.py: JWT (jose) + Argon2id password hashing
helpers are now private to this module.

logout is intentionally NOT exposed as a server endpoint — the
frontend simply discards the bearer token (PTD U02/U11/U18 are
boundary-only flows).
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from typing import Any

import psycopg2
from argon2 import PasswordHasher, Type
from argon2.exceptions import VerifyMismatchError
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from jose import JWTError, jwt
from pydantic import BaseModel


# ── JWT / hashing config (private) ───────────────────────────────────
_SECRET_KEY = os.getenv("JWT_SECRET", "fyp-demo-change-me")
_ALGORITHM = "HS256"
_TOKEN_TTL_HOURS = 12
_ph = PasswordHasher(type=Type.ID, memory_cost=65536, time_cost=3, parallelism=4)


@dataclass
class CurrentUser:
    account_id: int
    role: str
    email: str
    full_name: str | None = None


# ── password helpers (private) ───────────────────────────────────────
def _hash_password(raw: str) -> str:
    return _ph.hash(raw)


def _verify_password(raw: str, hashed: str) -> bool:
    try:
        _ph.verify(hashed, raw)
        return True
    except (VerifyMismatchError, Exception):
        return False


def _needs_rehash(hashed: str) -> bool:
    try:
        return _ph.check_needs_rehash(hashed)
    except Exception:
        return True


# ── token helpers (private) ──────────────────────────────────────────
def _create_token(account_id: int, role: str, email: str) -> str:
    payload = {
        "sub": str(account_id),
        "role": role,
        "email": email,
        "exp": dt.datetime.utcnow() + dt.timedelta(hours=_TOKEN_TTL_HOURS),
    }
    return jwt.encode(payload, _SECRET_KEY, algorithm=_ALGORITHM)


def _decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, _SECRET_KEY, algorithms=[_ALGORITHM])
    except JWTError as exc:
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}")


# ── FastAPI dependencies (public — imported by other core modules) ──
def get_current_user(
    request: Request, authorization: str | None = Header(default=None)
) -> CurrentUser:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    data = _decode_token(token)
    account_id = int(data["sub"])
    # U20: deactivation must take effect immediately, so a still-valid JWT
    # is not enough — the account row must exist and be active.
    with psycopg2.connect(request.app.state.cfg.database_url) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT status FROM user_account WHERE accountid = %s", (account_id,)
        )
        row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=401, detail="Account no longer exists")
    if row[0] != "active":
        raise HTTPException(status_code=403, detail="This account has been deactivated")
    return CurrentUser(
        account_id=account_id,
        role=data.get("role", "student"),
        email=data.get("email", ""),
    )


def require_role(*roles: str):
    def _dep(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if user.role not in roles:
            raise HTTPException(status_code=403, detail="Forbidden")
        return user
    return _dep


# ── Class ────────────────────────────────────────────────────────────
class UserInformation:
    """Business entity for user accounts and credentials."""

    def __init__(self, database_url: str):
        self.database_url = database_url

    # U01 / U10 / U17
    def login(self, email: str, password: str) -> dict[str, Any]:
        user = self.validateUserAccount(email, password)
        if not user:
            raise HTTPException(status_code=401, detail="Incorrect email or password")
        token = _create_token(user.account_id, user.role, user.email)
        return {
            "success": True,
            "token": token,
            "user": {
                "account_id": user.account_id,
                "role": user.role,
                "email": user.email,
                "full_name": user.full_name,
            },
        }

    # Helper used by login — DB lookup + password verify + transparent rehash
    def validateUserAccount(self, email: str, password: str) -> CurrentUser | None:
        sql = """
            SELECT ua.accountid, up.role, ua.email, ua.password_hash, pi.full_name,
                   ua.status
            FROM user_account ua
            JOIN user_profiles up ON up.profileid = ua.profileid
            LEFT JOIN personal_info pi ON pi.accountid = ua.accountid
            WHERE lower(ua.email) = lower(%s)
            LIMIT 1
        """
        with psycopg2.connect(self.database_url) as conn, conn.cursor() as cur:
            cur.execute(sql, (email,))
            row = cur.fetchone()
        if not row:
            return None
        account_id, role, em, pw_hash, full_name, status = row
        if not _verify_password(password, pw_hash):
            return None
        if status != "active":
            raise HTTPException(status_code=403, detail="This account has been deactivated")
        if _needs_rehash(pw_hash):
            try:
                with psycopg2.connect(self.database_url) as conn, conn.cursor() as cur:
                    cur.execute(
                        "UPDATE user_account SET password_hash = %s WHERE accountid = %s",
                        (_hash_password(password), account_id),
                    )
            except Exception:
                pass
        return CurrentUser(account_id=account_id, role=role, email=em, full_name=full_name)

    # U20: manage information (admin CRUD on user accounts)
    def manageInformation(
        self,
        action: str,
        actor: CurrentUser,
        payload: dict[str, Any] | None = None,
        target_account_id: int | None = None,
    ) -> dict[str, Any]:
        """Dispatch user-management actions for admins (U20)
        and self-update for any logged-in user.

        Supported actions:
          - "list"             admin only — list all users
          - "create"           admin only — create student/teacher/admin
          - "set_status"       admin only — activate / deactivate
          - "view_self"        any role — view own profile
        """
        if action == "list":
            self._require_admin(actor)
            return {"success": True, "users": self._list_users()}

        if action == "create":
            self._require_admin(actor)
            return self._create_user(payload or {})

        if action == "set_status":
            self._require_admin(actor)
            if target_account_id is None:
                raise HTTPException(400, "Missing target_account_id")
            return self._set_user_status(target_account_id, (payload or {}).get("status", ""))

        if action == "update":
            self._require_admin(actor)
            if target_account_id is None:
                raise HTTPException(400, "Missing target_account_id")
            return self._update_user(target_account_id, payload or {})

        if action == "view_self":
            return {
                "account_id": actor.account_id,
                "role": actor.role,
                "email": actor.email,
            }

        raise HTTPException(400, f"Unknown action: {action}")

    # ── internals ────────────────────────────────────────────────────
    @staticmethod
    def _require_admin(actor: CurrentUser) -> None:
        if actor.role != "admin":
            raise HTTPException(403, "Administrator privileges required")

    def _list_users(self) -> list[dict[str, Any]]:
        sql = """
            SELECT ua.accountid, ua.email, up.role, ua.status,
                   pi.full_name, pi.student_id, pi.staff_id, ua.created_at
            FROM user_account ua
            JOIN user_profiles up ON up.profileid = ua.profileid
            LEFT JOIN personal_info pi ON pi.accountid = ua.accountid
            ORDER BY ua.accountid
        """
        with psycopg2.connect(self.database_url) as conn, conn.cursor() as cur:
            cur.execute(sql)
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    def _create_user(self, body: dict[str, Any]) -> dict[str, Any]:
        role = body.get("role")
        if role not in ("student", "teacher", "admin"):
            raise HTTPException(400, "Invalid role")
        email = body.get("email")
        password = body.get("password")
        full_name = body.get("full_name")
        if not (email and password and full_name):
            raise HTTPException(400, "email / password / full_name are required")

        with psycopg2.connect(self.database_url) as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT profileid FROM user_profiles WHERE role = %s LIMIT 1",
                        (role,),
                    )
                    prow = cur.fetchone()
                    if not prow:
                        raise HTTPException(500, f"Role profile not found: {role}")
                    profile_id = prow[0]

                    cur.execute(
                        """
                        INSERT INTO user_account (profileid, email, password_hash)
                        VALUES (%s, %s, %s) RETURNING accountid
                        """,
                        (profile_id, email, _hash_password(password)),
                    )
                    account_id = cur.fetchone()[0]
                    student_id = body.get("student_id") or (
                        None if role != "student" else f"S{account_id:05d}"
                    )
                    staff_id = body.get("staff_id") or (
                        None if role == "student" else f"A{account_id:05d}"
                    )
                    cur.execute(
                        """
                        INSERT INTO personal_info (accountid, full_name, student_id, staff_id)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (account_id, full_name, student_id, staff_id),
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return {"success": True, "account_id": account_id}

    def _update_user(self, account_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        """Admin edit of a user's basic details (U20). Only the supplied
        fields are changed: email (user_account) and full_name / student_id /
        staff_id (personal_info)."""
        email = payload.get("email")
        full_name = payload.get("full_name")
        student_id = payload.get("student_id")
        staff_id = payload.get("staff_id")
        with psycopg2.connect(self.database_url) as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1 FROM user_account WHERE accountid = %s", (account_id,))
                    if not cur.fetchone():
                        raise HTTPException(404, "User not found")
                    if email:
                        cur.execute(
                            "SELECT 1 FROM user_account WHERE email = %s AND accountid <> %s",
                            (email, account_id),
                        )
                        if cur.fetchone():
                            raise HTTPException(409, f"Email {email} is already in use")
                        cur.execute(
                            "UPDATE user_account SET email = %s WHERE accountid = %s",
                            (email, account_id),
                        )
                    cur.execute("SELECT 1 FROM personal_info WHERE accountid = %s", (account_id,))
                    if cur.fetchone():
                        sets, params = [], []
                        if full_name is not None:
                            sets.append("full_name = %s")
                            params.append(full_name)
                        if student_id is not None:
                            sets.append("student_id = %s")
                            params.append(student_id or None)
                        if staff_id is not None:
                            sets.append("staff_id = %s")
                            params.append(staff_id or None)
                        if sets:
                            params.append(account_id)
                            cur.execute(
                                f"UPDATE personal_info SET {', '.join(sets)} WHERE accountid = %s",
                                params,
                            )
                    else:
                        cur.execute(
                            """INSERT INTO personal_info (accountid, full_name, student_id, staff_id)
                               VALUES (%s, %s, %s, %s)""",
                            (account_id, full_name, student_id or None, staff_id or None),
                        )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return {"success": True}

    def _set_user_status(self, account_id: int, status: str) -> dict[str, Any]:
        if status not in ("active", "inactive"):
            raise HTTPException(400, "Invalid status")
        with psycopg2.connect(self.database_url) as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE user_account SET status = %s, updated_at = NOW() "
                        "WHERE accountid = %s",
                        (status, account_id),
                    )
                    if cur.rowcount == 0:
                        raise HTTPException(404, "User not found")
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return {"success": True}


# ── Router ───────────────────────────────────────────────────────────
router = APIRouter(tags=["userInformation"])


def _svc(request: Request) -> UserInformation:
    return UserInformation(request.app.state.cfg.database_url)


class LoginBody(BaseModel):
    email: str
    password: str


@router.post("/auth/login")
def login_endpoint(body: LoginBody, request: Request):
    return _svc(request).login(body.email, body.password)


@router.get("/auth/me")
def me_endpoint(user: CurrentUser = Depends(get_current_user)):
    return {"account_id": user.account_id, "role": user.role, "email": user.email}


# Admin U20 — manage user accounts
class CreateUserBody(BaseModel):
    email: str
    password: str
    role: str
    full_name: str
    student_id: str | None = None
    staff_id: str | None = None


class StatusBody(BaseModel):
    status: str


class UpdateUserBody(BaseModel):
    email: str | None = None
    full_name: str | None = None
    student_id: str | None = None
    staff_id: str | None = None


@router.get("/admin/users")
def admin_list_users(
    request: Request,
    user: CurrentUser = Depends(require_role("admin")),
):
    return {"success": True, "users": _svc(request)._list_users()}


@router.post("/admin/users")
def admin_create_user(
    body: CreateUserBody,
    request: Request,
    user: CurrentUser = Depends(require_role("admin")),
):
    return _svc(request).manageInformation("create", user, body.model_dump())


@router.patch("/admin/users/{account_id}/status")
def admin_set_user_status(
    account_id: int,
    body: StatusBody,
    request: Request,
    user: CurrentUser = Depends(require_role("admin")),
):
    return _svc(request).manageInformation(
        "set_status", user, body.model_dump(), target_account_id=account_id
    )


@router.patch("/admin/users/{account_id}")
def admin_update_user(
    account_id: int,
    body: UpdateUserBody,
    request: Request,
    user: CurrentUser = Depends(require_role("admin")),
):
    return _svc(request).manageInformation(
        "update", user, body.model_dump(exclude_none=True), target_account_id=account_id
    )
