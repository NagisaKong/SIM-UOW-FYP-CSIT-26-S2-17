"""Prototype demo runner -- exercise the full AI pipeline end to end.

Modes
-----
* ``--ping``                    verify Supabase + model load only
* ``--list-cameras``            probe camera indices 0-9 and exit
* ``--query PATH``              batch-match still images, save annotated outputs
* ``--webcam``                  live inference (default camera, index 0)
* ``--webcam --camera N``       specify camera index (virtual cameras etc.)
* ``--webcam --video FILE``     use a video file or RTSP stream

Examples
--------
    python -m ai.run_prototype --ping
    python -m ai.run_prototype --list-cameras
    python -m ai.run_prototype --webcam --camera 1
    python -m ai.run_prototype --webcam --video test.mp4
    python -m ai.run_prototype --query ./test_images --save ./output
    python -m ai.run_prototype --webcam --course 1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2

from .config import AIConfig
from .attendance_pipeline import AttendancePipeline


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ── Argument parsing ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FYP-26-S2-17 attendance AI prototype runner",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--ping",         action="store_true", help="Check DB + model load only")
    mode.add_argument("--list-cameras", action="store_true", help="Probe camera indices 0-9")
    mode.add_argument("--webcam",       action="store_true", help="Live camera / video inference")
    mode.add_argument("--query",        metavar="PATH",      help="Still image or folder")

    p.add_argument("--camera", type=int, default=None, metavar="N",
                   help="Camera index for --webcam (default 0). "
                        "Virtual cameras are usually 1, 2 ...")
    p.add_argument("--video",  metavar="FILE",
                   help="Video file or RTSP URL instead of a live camera.")
    p.add_argument("--course", type=int, metavar="ID",
                   help="Write attendance records into this course's active session.")
    p.add_argument("--save",   default="output", metavar="DIR",
                   help="Output folder for annotated images (default: ./output)")
    return p.parse_args()


# ── Camera probe ──────────────────────────────────────────────────────────────

def _list_cameras(max_index: int = 9) -> None:
    """Try every camera index and print which ones open successfully."""
    print("Probing camera indices 0-{} ...".format(max_index))
    found = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  [{i}]  OK  {w}x{h}")
            found.append(i)
            cap.release()
        else:
            cap.release()
            print(f"  [{i}]  --  not available")
    if found:
        print(f"\nUse --camera {found[0]} (or any index above) with --webcam.")
    else:
        print("\nNo cameras found. Is a virtual camera (OBS, ManyCam ...) running?")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _collect_images(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        return sorted(
            f.path
            for f in os.scandir(path)
            if f.is_file() and os.path.splitext(f.name)[1].lower() in IMAGE_EXTS
        )
    return []


def _print_predictions(filename: str, result) -> None:
    if not result.predictions:
        print(f"  [{filename}]  no face detected")
        return
    tag_enh = "[enh]" if result.enhanced else "     "
    for pred in result.predictions:
        tag = "MATCH   " if pred.recognised else "UNKNOWN "
        sid = pred.student_id or (f"acc#{pred.account_id}" if pred.account_id else "-")
        per = "  ".join(
            f"{name}={trace['score']:.2f}"
            for name, trace in pred.per_model.items()
        )
        print(
            f"  {tag_enh} [{filename}]  {tag}  student={sid:<12s}"
            f"  fused={pred.score:.3f}  det={pred.det_score:.3f}  [{per}]"
        )


# ── Open capture (camera or file) ─────────────────────────────────────────────

def _open_capture(camera: int | None, video: str | None) -> cv2.VideoCapture:
    if video:
        cap = cv2.VideoCapture(video)
        src = video
    else:
        idx = camera if camera is not None else 0
        src = f"camera index {idx}"
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open {src}.\n"
            "  Run --list-cameras to see available indices.\n"
            "  Make sure OBS / ManyCam is active before launching."
        )
    return cap


# ── Inference modes ───────────────────────────────────────────────────────────

def _run_query(pipeline: AttendancePipeline, query: str, save_dir: str) -> int:
    images = _collect_images(query)
    if not images:
        print(f"ERROR: no images found at '{query}'")
        return 1

    os.makedirs(save_dir, exist_ok=True)
    print(f"\n-- Batch recognition ({len(images)} image(s)) --")

    for path in images:
        frame = cv2.imread(path)
        if frame is None:
            print(f"  [{os.path.basename(path)}]  could not read file")
            continue
        result = pipeline.process_frame(frame)
        _print_predictions(os.path.basename(path), result)
        annotated = pipeline.draw(frame, result)
        cv2.imwrite(os.path.join(save_dir, os.path.basename(path)), annotated)

    print(f"\nAnnotated outputs -> {Path(save_dir).resolve()}")
    return 0


def _run_webcam(
    pipeline: AttendancePipeline,
    camera: int | None,
    video: str | None,
    course_id: int | None,
) -> int:
    try:
        cap = _open_capture(camera, video)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    src_label = video or f"camera {camera if camera is not None else 0}"
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 0
    print(f"[webcam] source={src_label}  {w}x{h}  {fps:.1f} fps")

    session_id: int | None = None
    marked: set[int] = set()
    if course_id is not None:
        session_id = pipeline.store_manager.repo.get_or_open_session(course_id)
        print(f"[webcam] attendance session #{session_id} (course {course_id})")

    print("Running -- press  Q  in the window to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result  = pipeline.process_frame(frame)
            display = pipeline.draw(frame, result)
            cv2.imshow(f"Attendance  [{src_label}]  Q=quit", display)

            if session_id is not None:
                for pred in result.predictions:
                    if pred.recognised and pred.account_id not in marked:
                        pipeline.store_manager.repo.mark_attendance(
                            session_id=session_id,
                            account_id=pred.account_id,
                            status="present",
                        )
                        marked.add(pred.account_id)
                        name = pred.student_id or pred.full_name or f"acc#{pred.account_id}"
                        print(f"[attendance] {name} marked present")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    args = _parse_args()

    if args.list_cameras:
        _list_cameras()
        return 0

    cfg = AIConfig()
    print(cfg.log_summary())

    if args.ping:
        from .database_manager import EmbeddingRepo
        repo = EmbeddingRepo(cfg.database_url)
        print("[ping] connecting to Supabase...", end=" ", flush=True)
        repo.ping()
        print("ok")
        print("[ping] loading models...", flush=True)
        pipeline = AttendancePipeline.from_env(cfg)
        stores_info = {k: len(v) for k, v in pipeline.store_manager.stores.items()}
        print(f"[ping] pipeline ready  stores={stores_info}")
        return 0

    pipeline = AttendancePipeline.from_env(cfg)

    if args.webcam:
        return _run_webcam(pipeline, args.camera, args.video, args.course)

    if args.query:
        return _run_query(pipeline, args.query, args.save)

    print("ERROR: specify --ping | --list-cameras | --webcam | --query PATH")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[run_prototype] interrupted.")
        sys.exit(130)
