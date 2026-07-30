"""Class 4: facialImage  (business class — thin wrapper over the
AttendancePipeline service class in core/attendancePipeline.py).

Covers UC: U06 Student Facial Image Registration,
            U09 Update Student Facial Image,
            U19 Register Student with Facial Image (admin bulk),
            U21 Manage Facial Image Database (CRUD).

All AI inference primitives (detectors, recognizers, enhancers,
ensemble voting, embedding stores, the pipeline class itself, plus the
AIConfig and EmbeddingRepo helpers) now live in
core/attendancePipeline.py — see the docstring there for the layout.
"""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np
import psycopg2
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from core.attendancePipeline import AttendancePipeline, MultipleFacesError
from core.userInformation import CurrentUser, get_current_user, require_role


async def _bytes_to_cv2(file: UploadFile) -> np.ndarray:
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Unable to parse image file")
    return img


# ── Image-quality validation (UCD: «include» Validate Image Quality) ──
# Enrolment quality gate applied before any embedding is computed. A blurred,
# under/over-exposed or tiny photo yields a weak embedding that silently
# degrades recognition for that student, so it is rejected up front with an
# actionable message instead. Thresholds are env-tunable.
_MIN_IMAGE_PX = int(os.getenv("AI_ENROL_MIN_IMAGE_PX", "96"))
_MIN_SHARPNESS = float(os.getenv("AI_ENROL_MIN_SHARPNESS", "60"))
_MIN_BRIGHTNESS = float(os.getenv("AI_ENROL_MIN_BRIGHTNESS", "40"))
_MAX_BRIGHTNESS = float(os.getenv("AI_ENROL_MAX_BRIGHTNESS", "225"))


def assess_image_quality(image: np.ndarray) -> tuple[bool, str, dict[str, float]]:
    """Return (ok, message, metrics) for an enrolment photo.

    Metrics: `sharpness` is the variance of the Laplacian (low = blurred);
    `brightness` is the mean grey level (low = under-exposed, high = washed
    out / back-lit); `min_side` is the shorter image edge in pixels.
    """
    h, w = image.shape[:2]
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    brightness = float(grey.mean())
    metrics = {
        "sharpness": round(sharpness, 1),
        "brightness": round(brightness, 1),
        "min_side": float(min(h, w)),
    }
    if min(h, w) < _MIN_IMAGE_PX:
        return False, (
            f"Photo is too small ({w}x{h}). Please upload an image at least "
            f"{_MIN_IMAGE_PX}x{_MIN_IMAGE_PX} pixels."
        ), metrics
    if sharpness < _MIN_SHARPNESS:
        return False, (
            "Photo looks blurred. Please hold the camera steady and retake."
        ), metrics
    if brightness < _MIN_BRIGHTNESS:
        return False, (
            "Photo is too dark. Please move to a better-lit place and retake."
        ), metrics
    if brightness > _MAX_BRIGHTNESS:
        return False, (
            "Photo is over-exposed. Please avoid strong backlight and retake."
        ), metrics
    return True, "", metrics


# ════════════════════════════════════════════════════════════════════════
# Business class — FacialImage (per FYP class diagram).
# ════════════════════════════════════════════════════════════════════════
class FacialImage:
    """Face-enrollment business entity.

    Delegates the heavy lifting to AttendancePipeline (Service Class).
    """

    def __init__(self, pipeline: AttendancePipeline, database_url: str):
        self.pipeline = pipeline
        self.database_url = database_url

    # ── U09: student updates their own face ──────────────────────────
    def updateFacialImageForMe(
        self, account_id: int, image: np.ndarray,
    ) -> dict[str, Any]:
        return self._enrol(account_id, [image])

    # ── U06: student self-enrol (alias as per class diagram) ─────────
    def registerFacialImageForMe(
        self, account_id: int, image: np.ndarray,
    ) -> dict[str, Any]:
        return self._enrol(account_id, [image])

    # ── U19: admin registers anyone ──────────────────────────────────
    def registerFacialImageForStu(
        self, account_id: int, image: np.ndarray,
    ) -> dict[str, Any]:
        return self._enrol(account_id, [image])

    # ── U21: admin CRUD on face_embedding ────────────────────────────
    def manageFacialImageForStu(self, action: str, **kwargs) -> dict[str, Any]:
        if action == "list":
            return {"success": True, "faces": self._list_faces()}
        if action == "delete":
            face_id = kwargs.get("face_id")
            if face_id is None:
                raise HTTPException(400, "Missing face_id")
            return self._soft_delete_face(int(face_id))
        raise HTTPException(400, f"Unknown action: {action}")

    # ── Optional bulk helper (replaces the old ai/enrol.py CLI) ──────
    def bulk_enrol_from_folder(
        self, folder: str, dry_run: bool = False,
    ) -> dict[str, str]:
        """Process <folder>/<student_id>/*.jpg subdirectories.
        Returns {student_id: message}."""
        results: dict[str, str] = {}
        if not os.path.isdir(folder):
            return {"error": f"{folder} is not a directory"}
        image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        repo = self.pipeline.store_manager.repo
        for entry in sorted(os.scandir(folder), key=lambda e: e.name):
            if not entry.is_dir():
                continue
            sid = entry.name
            paths = sorted(
                f.path for f in os.scandir(entry.path)
                if f.is_file() and os.path.splitext(f.name)[1].lower() in image_exts
            )
            account_id = repo.find_account_by_student_id(sid)
            if account_id is None:
                results[sid] = f"skipped (no account for student_id={sid})"
                continue
            images = [img for img in (cv2.imread(p) for p in paths) if img is not None]
            if not images:
                results[sid] = "skipped (no readable images)"
                continue
            if dry_run:
                result = self.pipeline.process_frame(images[0])
                results[sid] = (
                    f"dry-run ok ({len(images)} photos, "
                    f"{len(result.predictions)} faces in first)"
                )
                continue
            written = self.pipeline.enrol_student(account_id=account_id, images=images)
            if not written:
                results[sid] = "skipped (no face detected)"
            else:
                summary = ", ".join(f"{m}->faceid={fid}" for m, fid in written.items())
                results[sid] = f"ok ({len(images)} photos; {summary})"
        return results

    # ── internals ────────────────────────────────────────────────────
    def _enrol(
        self, account_id: int, images: list[np.ndarray],
    ) -> dict[str, Any]:
        # «include» Validate Image Quality — reject unusable photos before
        # any embedding is written (see assess_image_quality above). The
        # single-face rule below is the second half of the same validation.
        for img in images:
            ok, message, metrics = assess_image_quality(img)
            if not ok:
                return {"success": False, "message": message, "quality": metrics}
        try:
            written = self.pipeline.enrol_student(
                account_id=account_id, images=images, reject_multiple=True,
            )
        except MultipleFacesError as e:
            return {
                "success": False,
                "message": (
                    f"{e.count} faces detected. Please make sure only your face "
                    "is in the photo and retake."
                ),
            }
        if not written:
            return {"success": False, "message": "No face detected, please retake the photo"}
        return {
            "success": True,
            "message": f"Account {account_id} enrolled successfully",
            "written": written,
        }

    def _list_faces(self) -> list[dict[str, Any]]:
        sql = """
            SELECT f.faceid, f.accountid, pi.full_name, pi.student_id,
                   f.model_name, f.model_version, f.dimension, f.is_active,
                   f.created_at
            FROM face_embedding f
            LEFT JOIN personal_info pi ON pi.accountid = f.accountid
            ORDER BY f.accountid, f.model_name, f.created_at DESC
        """
        with psycopg2.connect(self.database_url) as conn, conn.cursor() as cur:
            cur.execute(sql)
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    def _soft_delete_face(self, face_id: int) -> dict[str, Any]:
        with psycopg2.connect(self.database_url) as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE face_embedding SET is_active = FALSE WHERE faceid = %s",
                        (face_id,),
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        try:
            self.pipeline.store_manager.reload()
        except Exception:
            pass
        return {"success": True}


# ════════════════════════════════════════════════════════════════════════
# Router
# ════════════════════════════════════════════════════════════════════════
router = APIRouter(tags=["facialImage"])


def _svc(request: Request) -> FacialImage:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Face recognition is unavailable (AI pipeline failed to load)",
        )
    return FacialImage(
        pipeline,
        request.app.state.cfg.database_url,
    )


@router.post("/register")
async def register_face(
    request: Request,
    account_id: int = Form(...),
    file: UploadFile = File(...),
    user: CurrentUser = Depends(get_current_user),
):
    """U06/U09/U19: register or update a face image.
    Students may only register their own face; admins may register anyone."""
    if user.role != "admin" and user.account_id != account_id:
        raise HTTPException(status_code=403, detail="You may only enrol your own face")
    img = await _bytes_to_cv2(file)
    svc = _svc(request)
    if user.role == "admin" and user.account_id != account_id:
        return svc.registerFacialImageForStu(account_id, img)
    return svc.updateFacialImageForMe(account_id, img)


@router.post("/identify")
async def identify_face(
    request: Request,
    file: UploadFile = File(...),
    user: CurrentUser = Depends(get_current_user),
):
    """Generic face identification (used by demo + admin tools)."""
    img = await _bytes_to_cv2(file)
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Face recognition is unavailable (AI pipeline failed to load)",
        )
    result = pipeline.process_frame(img)
    identities = []
    for p in result.predictions:
        label = (
            p.full_name or p.student_id or (f"acc#{p.account_id}" if p.account_id else "Unknown")
            if p.recognised else "Unknown"
        )
        identities.append({
            "name": label,
            "confidence": round(float(p.score), 2),
            "recognised": p.recognised,
            "account_id": p.account_id,
            "student_id": p.student_id,
            "full_name": p.full_name,
            "score": round(float(p.score), 4),
            "det_score": round(float(p.det_score), 4),
            "bbox": list(p.bbox),
        })
    return {"success": True, "enhanced": result.enhanced, "identities": identities}


@router.get("/admin/faces")
def admin_list_faces(
    request: Request, user: CurrentUser = Depends(require_role("admin")),
):
    return _svc(request).manageFacialImageForStu("list")


@router.delete("/admin/faces/{face_id}")
def admin_delete_face(
    face_id: int,
    request: Request,
    user: CurrentUser = Depends(require_role("admin")),
):
    return _svc(request).manageFacialImageForStu("delete", face_id=face_id)
