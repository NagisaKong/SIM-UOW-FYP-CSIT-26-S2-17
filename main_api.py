"""FastAPI entrypoint for the FYP-26-S2-17 attendance system.

Loads the AttendancePipeline once at startup, mounts every business-class
router from core/, and starts uvicorn.

Run:
    python main_api.py
    # or:
    uvicorn main_api:app --host 127.0.0.1 --port 8001
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Allow running this file directly (`python main_api.py`).
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import os

import psycopg2
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core import all_routers
from core.attendancePipeline import AIConfig, AttendancePipeline
from core.attendanceSession import purge_expired_recordings


# ── Connection timezone patch ─────────────────────────────────────────
# Supabase's connection pooler (Supavisor) ignores database/role-level
# `ALTER ... SET timezone` and startup `options`, so pooled sessions always
# come up as UTC. That makes timestamps render in UTC everywhere they are
# stringified directly (notification emails, CSV report export, …).
#
# A per-session `SET TIME ZONE` is honoured by the pooler, so we wrap
# psycopg2.connect once here to apply it to every connection the app opens
# (every module calls psycopg2.connect directly). Configurable via
# APP_TIMEZONE; defaults to the demo/deployment locale.
_APP_TIMEZONE = os.getenv("APP_TIMEZONE", "Asia/Singapore")
_orig_pg_connect = psycopg2.connect


def _connect_with_timezone(*args, **kwargs):
    conn = _orig_pg_connect(*args, **kwargs)
    try:
        prev_autocommit = conn.autocommit
        conn.autocommit = True  # make the SET stick at session level
        with conn.cursor() as cur:
            cur.execute("SET TIME ZONE %s", (_APP_TIMEZONE,))
        conn.autocommit = prev_autocommit
    except Exception as exc:  # noqa: BLE001 — never let tz setup break a connection
        print(f"[timezone] could not set session tz to {_APP_TIMEZONE}: {exc}")
    return conn


psycopg2.connect = _connect_with_timezone


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = AIConfig()
    print(cfg.log_summary())
    app.state.cfg = cfg
    app.state.pipeline = AttendancePipeline.from_env(cfg)
    # U03 retention: drop class recordings / detection rows past their 30-day expiry.
    try:
        stats = purge_expired_recordings(cfg.database_url)
        if stats["recordings_deleted"]:
            print(f"[retention] purged {stats['recordings_deleted']} expired recording(s)")
    except Exception as exc:  # noqa: BLE001 — startup must not crash on cleanup
        print(f"[retention] skipped: {exc}")
    yield


app = FastAPI(
    title="SIM-UOW Face Attendance System API",
    description=(
        "Face enrolment + identification + role-scoped endpoints, "
        "organised by the 9 business classes in core/."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "service": "SIM-UOW Face Attendance API",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
        "frontend": "http://127.0.0.1:5500",
    }


@app.get("/health")
def health():
    pipeline: AttendancePipeline = app.state.pipeline
    stores = {name: len(s) for name, s in pipeline.store_manager.stores.items()}
    return {"success": True, "stores": stores}


# Mount every business-class router (userInformation, attendanceRecord,
# notification, facialImage, attendanceSession, attendanceAppeal, report,
# inClassBehaviour, trainConfiguration).
for r in all_routers:
    app.include_router(r)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
