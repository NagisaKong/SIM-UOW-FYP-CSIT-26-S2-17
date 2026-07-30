"""FastAPI entrypoint for the FYP-26-S2-17 attendance system.

Loads the AttendancePipeline once at startup, mounts every business-class
router from core/, and starts uvicorn.

Run:
    python main_api.py
    # or:
    uvicorn main_api:app --host 127.0.0.1 --port 8000
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
from core.attendanceSession import expire_overdue_sessions, purge_expired_recordings

# ── Connection timezone patch ─────────────────────────────────────────
# Supabase's pooler (Supavisor) ignores database-level `SET timezone` and
# startup options, so pooled sessions come up as UTC and every directly
# stringified timestamp (emails, CSV export) renders in UTC. A per-session
# `SET TIME ZONE` is honoured, so psycopg2.connect is wrapped once here to
# apply it to every connection the app opens.
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
    # Face models are heavy and env-sensitive (CUDA / onnxruntime / numpy).
    # Leave, appeal, auth, and attendance CRUD must still come up if the
    # AI stack fails to import — otherwise the local UI only shows
    # "Cannot reach the API at http://127.0.0.1:8000".
    try:
        app.state.pipeline = AttendancePipeline.from_env(cfg)
    except Exception as exc:  # noqa: BLE001 — keep non-AI routes available
        print(f"[pipeline] AI models failed to load: {exc}")
        print("[pipeline] running without face recognition (leave/appeal/auth OK)")
        app.state.pipeline = None
    # CR-06 behaviour analysis (drowsiness / phone / heatmap). Opt-in via
    # AI_BEHAVIOUR=true — detector models load lazily on the first frame.
    if cfg.behaviour_enabled and app.state.pipeline is not None:
        from core.behaviourAnalysis import BehaviourAnalysisService

        app.state.behaviour = BehaviourAnalysisService(cfg, cfg.database_url)
        print("[behaviour] CR-06 analysis service enabled (models load on first frame)")
    else:
        app.state.behaviour = None
    # U03 retention: drop class recordings / detection rows past their 30-day expiry.
    try:
        stats = purge_expired_recordings(cfg.database_url)
        if stats["recordings_deleted"]:
            print(f"[retention] purged {stats['recordings_deleted']} expired recording(s)")
    except Exception as exc:  # noqa: BLE001 — startup must not crash on cleanup
        print(f"[retention] skipped: {exc}")
    # U15: close out any session whose scheduled end passed while the server
    # was down, so nothing is still "in progress" once it comes back up.
    try:
        expire_overdue_sessions(cfg.database_url, force=True)
    except Exception as exc:  # noqa: BLE001 — startup must not crash on cleanup
        print(f"[expiry] startup sweep skipped: {exc}")
    yield


# ── Environment ───────────────────────────────────────────────────────
# APP_ENV controls production-only hardening (docs exposure, CORS, JWT).
_IS_PROD = os.getenv("APP_ENV", "development").lower() in {"production", "prod"}

# CORS: never use a wildcard with credentials. Restrict to the frontend
# origin(s). Configure via ALLOWED_ORIGINS (comma-separated) in production,
# e.g. "https://your-app.vercel.app". Defaults to the local dev frontend.
_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "ALLOWED_ORIGINS",
        "http://127.0.0.1:5050,http://localhost:5050,"
        "http://127.0.0.1:5500,http://localhost:5500",
    ).split(",")
    if o.strip()
]

app = FastAPI(
    title="SIM-UOW Face Attendance System API",
    description=(
        "Face enrolment + identification + role-scoped endpoints, "
        "organised by the 9 business classes in core/."
    ),
    version="3.0.0",
    lifespan=lifespan,
    # Hide interactive API docs / schema in production to reduce attack surface.
    docs_url=None if _IS_PROD else "/docs",
    redoc_url=None if _IS_PROD else "/redoc",
    openapi_url=None if _IS_PROD else "/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Security response headers ─────────────────────────────────────────
# Applied to every response. In production we send a strict policy (the API
# only returns JSON and docs are disabled); in development we relax CSP so
# the Swagger UI at /docs keeps working.
@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    # Stop MIME-sniffing, deny framing (clickjacking), limit referrer leakage,
    # and disable powerful browser features this API never needs.
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    if _IS_PROD:
        # HSTS only over HTTPS (Railway terminates TLS). 1 year + preload.
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains; preload"
        )
        # Lock the API down to its own origin; no framing at all.
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; frame-ancestors 'none'; base-uri 'none'"
        )
    return response


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
    """Liveness plus the AI configuration this instance is actually running.

    The ensemble is environment-dependent: the CPU cloud deployment runs
    ArcFace only (AI_USE_MTCNN/FACENET=false, see DEPLOY.md) while the GPU
    workstation runs the full dual-model ensemble and behaviour analysis.
    Reporting it here makes the tested configuration verifiable rather than
    assumed — test evidence can cite this payload.
    """
    pipeline: AttendancePipeline | None = getattr(app.state, "pipeline", None)
    cfg = app.state.cfg
    if pipeline is None:
        return {
            "success": True,
            "degraded": True,
            "stores": {},
            "recognition": None,
            "behaviour_analysis": False,
            "message": "API up; AI pipeline not loaded",
        }
    stores = {name: len(s) for name, s in pipeline.store_manager.stores.items()}
    active = [name for name, w in pipeline._weights.items() if w > 0]
    return {
        "success": True,
        "stores": stores,
        "recognition": {
            "detectors": ["scrfd"] + (["mtcnn"] if pipeline._mtcnn else []),
            "recognisers": active,
            "ensemble": len(active) >= 2,
            "voting_weights": dict(pipeline._weights),
            "device": cfg.device,
            "enhancer": pipeline.enhancer.name if pipeline.enhancer else None,
        },
        "behaviour_analysis": bool(getattr(app.state, "behaviour", None)),
    }


# Mount every business-class router (userInformation, attendanceRecord,
# notification, facialImage, attendanceSession, attendanceAppeal, report,
# behaviourAnalysis, trainConfiguration).
for r in all_routers:
    app.include_router(r)


if __name__ == "__main__":
    # Cloud platforms (Railway, etc.) inject the port via $PORT and require
    # binding to 0.0.0.0. Locally this still defaults to 127.0.0.1:8000.
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
