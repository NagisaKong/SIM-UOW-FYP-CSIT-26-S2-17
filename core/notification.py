"""Class 3: notification

Covers UC: U05 Attendance Status Notification (late/absent emails),
           U29 Long-term Absence Reminder,
           U08/U31 review-outcome emails (appeal / leave decisions).

U29 trigger thresholds come from the admin-configured
attendance_threshold_config table (U34); the module constants are only
the last-resort fallback when that table is unavailable.

Absorbs former api/notifications.py — SMTP helpers are now private
functions inside this module.
"""

from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage
from typing import Any, Iterable

import psycopg2
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request

from core.attendanceRecord import load_threshold_config
from core.userInformation import CurrentUser, require_role


# ── SMTP (private; absorbed from api/notifications.py) ───────────────
def _smtp_enabled() -> bool:
    return bool(os.getenv("SMTP_HOST"))


def _send_email(to_addr: str, subject: str, body: str) -> bool:
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
    except Exception as exc:
        print(f"[notify:error] {to_addr} | {subject} | {exc}")
        return False


# ── Class ────────────────────────────────────────────────────────────
class Notification:
    """Outbound notification entity."""

    # PTD's "absenceThreshold" lives on attendanceRecord; here we read it
    # via a simple default constant. configureAbsenceThreshold is in
    # attendanceRecord.py per the class diagram.
    DEFAULT_CONSECUTIVE_ABSENCE_THRESHOLD = 3
    DEFAULT_MIN_RATE = 70.0

    def __init__(self, database_url: str):
        self.database_url = database_url

    # U05: late/absent emails after a session ends
    def generateLateNotification(self, session_id: int) -> dict[str, Any]:
        recipients = self._fetch_late_absent_recipients(session_id)
        return _send_batch(recipients)

    # U29: long-term absence reminders (per-student check)
    def generateLongTermAbsenceNotification(
        self,
        consecutive_threshold: int | None = None,
        min_rate: float | None = None,
    ) -> dict[str, Any]:
        """Evaluate every active student; send a reminder when they hit
        either trigger. Approved leave applications are excluded.

        Thresholds default to the admin-configured values in
        attendance_threshold_config (U34); explicit arguments override.
        """
        cfg = load_threshold_config(self.database_url)
        n_thr = consecutive_threshold or int(
            cfg.get("absence_threshold", self.DEFAULT_CONSECUTIVE_ABSENCE_THRESHOLD)
        )
        rate_thr = min_rate if min_rate is not None else float(
            cfg.get("minimum_attendance_rate", self.DEFAULT_MIN_RATE)
        )

        # Pull each student's most recent records + overall rate.
        sql = """
            WITH per_student AS (
              SELECT r.accountid,
                     COUNT(*) FILTER (WHERE r.status IN ('present','late')) AS attended,
                     COUNT(*) AS total
              FROM attendance_record r
              JOIN attendance_session s ON s.attendancesessionid = r.attendancesessionid
              JOIN course_enrollment e
                ON e.accountid = r.accountid AND e.courseid = s.courseid
               AND e.status = 'active'
              WHERE s.status = 'ended'
                AND NOT EXISTS (
                  SELECT 1 FROM leave_application la
                  WHERE la.accountid = r.accountid
                    AND la.attendancesessionid = r.attendancesessionid
                    AND la.status = 'approved'
                )
              GROUP BY r.accountid
            )
            SELECT ua.accountid, ua.email, pi.full_name,
                   ps.attended, ps.total,
                   ROUND(ps.attended::numeric / NULLIF(ps.total,0) * 100, 1) AS rate
            FROM per_student ps
            JOIN user_account ua ON ua.accountid = ps.accountid
            LEFT JOIN personal_info pi ON pi.accountid = ps.accountid
            WHERE ua.status = 'active'
        """
        consec_sql = """
            SELECT r.accountid, COUNT(*) AS streak
            FROM (
              SELECT r.*,
                     ROW_NUMBER() OVER (PARTITION BY r.accountid
                                        ORDER BY s.start_time DESC) AS rn,
                     r.status AS st
              FROM attendance_record r
              JOIN attendance_session s ON s.attendancesessionid = r.attendancesessionid
              WHERE s.status = 'ended'
            ) r
            WHERE r.st = 'absent' AND r.rn <= %s
            GROUP BY r.accountid
            HAVING COUNT(*) >= %s
        """
        targets: dict[int, dict[str, Any]] = {}
        with psycopg2.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                cols = [c[0] for c in cur.description]
                for row in cur.fetchall():
                    rec = dict(zip(cols, row))
                    if rec["rate"] is not None and float(rec["rate"]) < rate_thr:
                        targets[rec["accountid"]] = rec
                cur.execute(consec_sql, (n_thr, n_thr))
                streaks = {r[0]: r[1] for r in cur.fetchall()}

        # Anyone whose latest N sessions are all absent is also a target.
        if streaks:
            with psycopg2.connect(self.database_url) as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT ua.accountid, ua.email, pi.full_name
                    FROM user_account ua
                    LEFT JOIN personal_info pi ON pi.accountid = ua.accountid
                    WHERE ua.accountid = ANY(%s)
                      AND ua.status = 'active'
                    """,
                    (list(streaks.keys()),),
                )
                for acc, email, full_name in cur.fetchall():
                    targets.setdefault(acc, {
                        "accountid": acc, "email": email, "full_name": full_name,
                        "attended": None, "total": None, "rate": None,
                    })

        sent, failed = 0, 0
        for rec in targets.values():
            if not rec.get("email"):
                continue
            subject = "[Attendance] Reminder: low attendance detected"
            body = (
                f"Hi {rec.get('full_name') or 'Student'},\n\n"
                f"Our records show your recent attendance has fallen below the "
                f"configured threshold (rate={rec.get('rate')}%, "
                f"consecutive_absences={streaks.get(rec['accountid'], 0)}).\n\n"
                "Please review your records on the student dashboard and "
                "contact your lecturer if there is a mistake.\n\n"
                "— FYP-26-S2-17 Attendance System"
            )
            if _send_email(rec["email"], subject, body):
                sent += 1
            else:
                failed += 1
        return {"success": True, "sent": sent, "failed": failed, "targets": len(targets)}

    # ── internal ─────────────────────────────────────────────────────
    def _fetch_late_absent_recipients(self, session_id: int) -> list[dict[str, Any]]:
        sql = """
            SELECT ua.email, pi.full_name, r.status,
                   c.course_code, c.course_name, s.start_time
            FROM attendance_record r
            JOIN attendance_session s ON s.attendancesessionid = r.attendancesessionid
            JOIN course c ON c.courseid = s.courseid
            JOIN user_account ua ON ua.accountid = r.accountid
            -- Only currently-enrolled, active accounts get emailed:
            -- skip students who have since withdrawn or been deactivated
            -- (a stale late/absent record must not trigger a notification).
            JOIN course_enrollment e
              ON e.accountid = r.accountid AND e.courseid = c.courseid
             AND e.status = 'active'
            LEFT JOIN personal_info pi ON pi.accountid = r.accountid
            WHERE r.attendancesessionid = %s
              AND r.status IN ('late', 'absent')
              AND ua.status = 'active'
        """
        with psycopg2.connect(self.database_url) as conn, conn.cursor() as cur:
            cur.execute(sql, (session_id,))
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]


# ── Public batch send (used as BackgroundTasks payload) ──────────────
def _send_batch(recipients: Iterable[dict]) -> dict[str, Any]:
    sent, failed = 0, 0
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
            "If you believe this is incorrect, please submit an appeal "
            "through the student dashboard, or contact your lecturer.\n\n"
            "— FYP-26-S2-17 Attendance System"
        )
        if _send_email(email, subject, body):
            sent += 1
        else:
            failed += 1
    return {"sent": sent, "failed": failed, "smtp_configured": _smtp_enabled()}


def send_late_absent_emails(recipients: Iterable[dict]) -> dict[str, Any]:
    """Backward-compatible export used by attendanceSession.endSession."""
    return _send_batch(recipients)


def send_review_outcome_email(recipient: dict[str, Any]) -> bool:
    """U08/U31: email a student the outcome of their appeal or leave
    application review. Expected keys: email, full_name, kind
    ('appeal' | 'leave application'), decision ('approved' | 'rejected'),
    course, start_time; optional: comment, corrected_status.
    Safe to run as a BackgroundTasks payload — never raises."""
    try:
        email = recipient.get("email")
        if not email:
            return False
        kind = recipient.get("kind", "request")
        decision = (recipient.get("decision") or "").upper()
        course = recipient.get("course") or ""
        when = recipient.get("start_time") or ""
        lines = [
            f"Hi {recipient.get('full_name') or 'Student'},",
            "",
            f"Your {kind} for {course} (session starting {when}) has been "
            f"reviewed: {decision}.",
        ]
        corrected = recipient.get("corrected_status")
        if corrected:
            lines.append(
                f"Your attendance record for this session has been updated "
                f"to: {corrected.upper()}."
            )
        comment = recipient.get("comment")
        if comment:
            lines.append(f"Reviewer comment: {comment}")
        lines += [
            "",
            "You can view the details on the student dashboard.",
            "",
            "— FYP-26-S2-17 Attendance System",
        ]
        subject = f"[Attendance] Your {kind} was {decision.lower()} — {course}"
        return _send_email(email, subject, "\n".join(lines))
    except Exception as exc:  # noqa: BLE001 — background task must not raise
        print(f"[notify:error] review outcome | {exc}")
        return False


# ── Router ───────────────────────────────────────────────────────────
router = APIRouter(tags=["notification"])


def _svc(request: Request) -> Notification:
    return Notification(request.app.state.cfg.database_url)


@router.post("/teacher/sessions/{session_id}/notify")
def teacher_resend_notifications(
    session_id: int,
    background_tasks: BackgroundTasks,
    request: Request,
    user: CurrentUser = Depends(require_role("teacher")),
):
    """U05 manual resend: re-email every late/absent student for this session."""
    with psycopg2.connect(request.app.state.cfg.database_url) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM attendance_session WHERE attendancesessionid = %s",
            (session_id,),
        )
        if not cur.fetchone():
            raise HTTPException(404, "Session not found")
    svc = _svc(request)
    recipients = svc._fetch_late_absent_recipients(session_id)
    background_tasks.add_task(_send_batch, recipients)
    return {"success": True, "queued": len(recipients)}


@router.post("/admin/notifications/long-term-absence")
def admin_trigger_long_term_absence(
    request: Request,
    background_tasks: BackgroundTasks,
    user: CurrentUser = Depends(require_role("admin")),
):
    """U29 manual trigger: scan all students and send reminders to those
    breaching the configured absence threshold."""
    svc = _svc(request)
    background_tasks.add_task(svc.generateLongTermAbsenceNotification)
    return {"success": True, "queued": True}
