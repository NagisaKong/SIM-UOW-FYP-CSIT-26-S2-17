"""FastAPI web entrypoint for the FYP-26-S2-17 attendance system.

Wraps `ai.attendance_pipeline.AttendancePipeline` (the full SCRFD + MTCNN +
ArcFace + FaceNet + GAN/CLAHE ensemble) behind HTTP endpoints so the
frontend can register students and run live identification.

Run:
    uvicorn api.main_api:app --host 127.0.0.1 --port 8000
    # or:
    python -m api.main_api
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running this file directly (e.g. `python api/main_api.py` or from an
# IDE's Run button). Adds the repo root to sys.path so `ai.*` resolves.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from contextlib import asynccontextmanager

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from ai.attendance_pipeline import AttendancePipeline
from ai.config import AIConfig


# ── Lifespan: load the pipeline once at startup ───────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = AIConfig()
    print(cfg.log_summary())
    app.state.pipeline = AttendancePipeline.from_env(cfg)
    yield


app = FastAPI(
    title="SIM-UOW Face Attendance System API",
    description="Face enrolment + identification backed by the ai.attendance_pipeline ensemble.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _bytes_to_cv2(file: UploadFile) -> np.ndarray:
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="无法解析图片文件")
    return img


@app.get("/health")
def health():
    pipeline: AttendancePipeline = app.state.pipeline
    stores = {name: len(s) for name, s in pipeline.store_manager.stores.items()}
    return {"success": True, "stores": stores}


@app.post("/register")
async def register_face(
    account_id: int = Form(...),
    file: UploadFile = File(...),
):
    """Extract embeddings from the uploaded photo and write them to Supabase."""
    pipeline: AttendancePipeline = app.state.pipeline
    img = await _bytes_to_cv2(file)

    written = pipeline.enrol_student(account_id=account_id, images=[img])
    if not written:
        return {"success": False, "message": "未检测到人脸，请重新拍摄"}
    return {
        "success": True,
        "message": f"学生 {account_id} 录入成功",
        "written": written,
    }


@app.post("/identify")
async def identify_face(file: UploadFile = File(...)):
    """Run the full ensemble pipeline on the uploaded frame."""
    pipeline: AttendancePipeline = app.state.pipeline
    img = await _bytes_to_cv2(file)

    result = pipeline.process_frame(img)

    identities = []
    for p in result.predictions:
        label = (
            p.full_name or p.student_id or (f"acc#{p.account_id}" if p.account_id else "Unknown")
            if p.recognised
            else "Unknown"
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

    return {
        "success": True,
        "enhanced": result.enhanced,
        "identities": identities,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
