"""Email notifications for U05 (late/absent alerts).

Stateless: Reads SMTP settings from env so the rest of the
codebase doesn't need to change. Designed to be called from FastAPI
BackgroundTasks so the HTTP response isn't blocked by SMTP latency.

Env vars:
    SMTP_HOST          required to actually send mail (otherwise we no-op + log)
    SMTP_PORT          default 587
    SMTP_USER          optional (omit for unauthenticated relays)
    SMTP_PASSWORD      optional
    SMTP_FROM          default = SMTP_USER or "noreply@fyp.local"
    SMTP_USE_TLS       "1" (default) to STARTTLS; "0" for plain
"""

from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage
from typing import Iterable


def _smtp_enabled() -> bool:
    return bool(os.getenv("SMTP_HOST"))


def _send(to_addr: str, subject: str, body: str) -> bool:
    """Send a single plain-text email. Returns True on success.

    No-ops (returns False, prints) if SMTP_HOST is unset, so dev environments
    without an SMTP relay still see the endpoint succeed.
    """
    if not _smtp_enabled():
        print(f"[notify:noop] {to_addr} | {subject}")
        return False

    host = os.environ["SMTP_HOST"]
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    pw = os.getenv("SMTP_PASSWORD")
    sender = os.getenv("SMTP_FROM") or user or "noreply@fyp.local"
    use_tls = os.getenv("SMTP_USE_TLS", "1") != "0"

    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        with smtplib.SMTP(host, port, timeout=15) as s:
            if use_tls:
                s.starttls(context=ssl.create_default_context())
            if user and pw:
                s.login(user, pw)
            s.send_message(msg)
        return True
    except Exception as exc:  # pragma: no cover - depends on external relay
        print(f"[notify:error] {to_addr} | {subject} | {exc}")
        return False


def send_late_absent_emails(recipients: Iterable[dict]) -> dict:
    """Email each recipient about their status for a session.

    recipients: iterable of {email, full_name, status, course_code,
                             course_name, start_time}
    """
    sent = 0
    failed = 0
    for r in recipients:
        status = (r.get("status") or "").lower()
        if status not in ("late", "absent"):
            continue
        email = r.get("email")
        if not email:
            continue
        name = r.get("full_name") or "Student"
        course = f"{r.get('course_code') or ''} {r.get('course_name') or ''}".strip()
        when = r.get("start_time") or ""
        subject = f"[Attendance] Marked {status.capitalize()} — {course}"
        body = (
            f"Hi {name},\n\n"
            f"Your attendance for {course} (session starting {when}) was "
            f"recorded as: {status.upper()}.\n\n"
            "If you believe this is incorrect, please submit an appeal through "
            "the student dashboard, or contact your lecturer.\n\n"
            "— FYP-26-S2-17 Attendance System"
        )
        if _send(email, subject, body):
            sent += 1
        else:
            failed += 1
    return {"sent": sent, "failed": failed, "smtp_configured": _smtp_enabled()}
