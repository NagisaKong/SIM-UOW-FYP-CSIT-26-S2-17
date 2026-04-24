"""One-click demo for the FYP-26-S2-17 attendance AI pipeline.

Run:
    python -m ai.demo
    # or via the batch launcher:
    demo.bat

Flow:
    1.  Check Supabase connectivity
    2.  Load AI models (SCRFD + ArcFace + optional MTCNN/FaceNet)
    3.  Hydrate in-memory embedding stores FROM THE DATABASE (no local files)
    4.  Run inference on test images or live camera

Face embeddings are loaded exclusively from the cloud Supabase database.
Use  python -m ai.enrol  to register new students.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2


# ── Repo-local paths ────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
ENROLMENT_DIR = REPO_ROOT / "ai" / "enrolment_images"
OUTPUT_DIR = REPO_ROOT / "output"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ── Pretty printing ─────────────────────────────────────────────────────────
def banner(title: str) -> None:
    line = "─" * max(12, len(title) + 4)
    print(f"\n{line}\n  {title}\n{line}")


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def warn(msg: str) -> None:
    print(f"  [warn] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


# ── Steps ───────────────────────────────────────────────────────────────────
def step_load_config():
    from .config import AIConfig

    cfg = AIConfig()
    print(cfg.log_summary())
    if not cfg.database_url:
        fail("DATABASE_URL missing — set it in .env before running the demo.")
        sys.exit(2)
    ok("Config loaded from .env")
    return cfg


def step_ping(cfg):
    from .db import EmbeddingRepo

    banner("Step 1/3  Connect to Supabase")
    repo = EmbeddingRepo(cfg.database_url)
    t0 = time.time()
    repo.ping()
    ok(f"Supabase reachable ({(time.time()-t0)*1000:.0f} ms)")
    return repo


def step_warm_models(cfg):
    from .pipeline import AttendancePipeline

    banner("Step 2/3  Load models + fetch embeddings from DB")
    t0 = time.time()
    pipeline = AttendancePipeline.from_env(cfg)
    elapsed = time.time() - t0

    total = sum(len(s) for s in pipeline.store_manager.stores.values())
    if total == 0:
        warn("No face embeddings found in the database.")
        warn("Run  python -m ai.enrol --folder <photos_dir>  to register students first.")
    else:
        ok(f"Pipeline ready ({elapsed:.1f}s)")

    for name, store in pipeline.store_manager.stores.items():
        status = "OK" if len(store) > 0 else "EMPTY"
        print(f"    [{status}] store[{name}]: {len(store)} embeddings  "
              f"(threshold={store.threshold})")
    return pipeline


def step_inference(pipeline, mode: str, camera: int = 0, video: str | None = None):
    banner(f"Step 4/5  Run inference  [{mode}]")

    if mode == "image":
        # Pick the first image we can find under enrolment_images or ./test_images
        candidates: list[Path] = []
        for base in (REPO_ROOT / "test_images", ENROLMENT_DIR):
            if base.exists():
                for p in base.rglob("*"):
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                        candidates.append(p)
        if not candidates:
            warn("No demo images found. Drop some into ./test_images/ and re-run.")
            return

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for path in candidates[:8]:  # cap noise
            frame = cv2.imread(str(path))
            result = pipeline.process_frame(frame)
            annotated = pipeline.draw(frame, result)
            out_path = OUTPUT_DIR / path.name
            cv2.imwrite(str(out_path), annotated)

            labels = [
                f"{p.student_id or 'Unknown'}(fused={p.score:.2f})"
                for p in result.predictions
            ]
            tag = "enhanced" if result.enhanced else "raw     "
            print(f"  [{tag}] {path.name:<25s}  faces={len(result.predictions)}  {labels}")
        ok(f"Annotated outputs → {OUTPUT_DIR}")
        return

    if mode == "webcam":
        src = video if video else camera
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW) if isinstance(src, int) else cv2.VideoCapture(src)
        if not cap.isOpened() and isinstance(src, int):
            cap = cv2.VideoCapture(src)   # retry without CAP_DSHOW
        if not cap.isOpened():
            fail(f"Cannot open source '{src}'. Run --list-cameras to find your virtual camera index.")
            return
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  Source: {src}  ({w}×{h})  — press Q in the window to stop.")
        try:
            while True:
                ok_, frame = cap.read()
                if not ok_:
                    break
                result = pipeline.process_frame(frame)
                cv2.imshow("FYP demo  [Q=quit]", pipeline.draw(frame, result))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


def step_summary(pipeline):
    banner("Step 3/3  Summary")
    for name, store in pipeline.store_manager.stores.items():
        print(f"  store[{name:8s}] threshold={store.threshold}  enrolled={len(store)}")
    print(f"  enhancer: {pipeline.enhancer.name if pipeline.enhancer else 'off'}")
    ok("Demo done.")


# ── Mode picker ─────────────────────────────────────────────────────────────
def pick_mode(default: str | None = None) -> str:
    """Prompt the user for image / webcam / skip. Honours --mode CLI arg."""
    if default:
        return default

    print()
    print("  Inference mode:")
    print("    [1] image   — run on sample/test images (no hardware needed)")
    print("    [2] webcam  — live camera inference")
    print("    [3] skip    — config/DB/model check only")
    choice = input("  Choose [1/2/3] (default 1): ").strip() or "1"
    return {"1": "image", "2": "webcam", "3": "skip"}.get(choice, "image")


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="FYP-26-S2-17 one-click demo")
    p.add_argument("--mode", choices=["image", "webcam", "skip"],
                   help="Skip the interactive prompt.")
    p.add_argument("--camera", type=int, default=0, metavar="N",
                   help="Camera index for webcam mode (default 0).")
    p.add_argument("--video", metavar="FILE",
                   help="Video file or RTSP URL instead of live camera.")
    p.add_argument("--list-cameras", action="store_true",
                   help="Probe camera indices 0-9 and exit.")
    # kept for backward-compat with demo.bat calls; has no effect now
    p.add_argument("--no-enrol", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.list_cameras:
        from .run_prototype import _list_cameras
        _list_cameras()
        return 0

    banner("FYP-26-S2-17  Attendance AI demo")
    cfg = step_load_config()
    step_ping(cfg)
    pipeline = step_warm_models(cfg)   # loads embeddings from DB only

    mode = pick_mode(args.mode)
    if mode != "skip":
        step_inference(pipeline, mode, camera=args.camera, video=args.video)

    step_summary(pipeline)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[demo] interrupted by user.")
        sys.exit(130)
