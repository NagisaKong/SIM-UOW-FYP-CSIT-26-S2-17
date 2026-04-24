"""Enrolment CLI — extract embeddings from enrolment photos and write them
to Supabase.

Expected folder layout (default: ./enrolment_images/):

    enrolment_images/
        <student_id>/
            photo1.jpg
            photo2.jpg
            ...

Each subfolder name must match `personal_info.student_id` already present
in the database. Run `database/schema.sql` + seed your users first.

Usage
-----
    python -m ai.enrol --folder ai/enrolment_images
    python -m ai.enrol --student S001 --images a.jpg b.jpg c.jpg
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2

from .config import AIConfig
from .db import EmbeddingRepo
from .pipeline import AttendancePipeline


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enrol students into Supabase.")
    p.add_argument("--folder", help="Folder with <student_id>/*.jpg subfolders")
    p.add_argument("--student", help="Single student_id to enrol")
    p.add_argument("--images", nargs="*", help="Image paths for --student mode")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract embeddings but do not write to the database",
    )
    return p.parse_args()


def _collect_images(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    return sorted(
        f.path
        for f in os.scandir(folder)
        if f.is_file() and os.path.splitext(f.name)[1].lower() in IMAGE_EXTS
    )


def _load_images(paths: list[str]):
    out = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"  ! could not read {p}")
            continue
        out.append(img)
    return out


def _enrol_one(
    pipeline: AttendancePipeline,
    repo: EmbeddingRepo,
    student_id: str,
    image_paths: list[str],
    dry_run: bool,
) -> str:
    account_id = repo.find_account_by_student_id(student_id)
    if account_id is None:
        return f"skipped (no account for student_id={student_id})"

    images = _load_images(image_paths)
    if not images:
        return "skipped (no readable images)"

    if dry_run:
        # Run the first image through the pipeline to confirm a face is
        # detectable — but don't write.
        result = pipeline.process_frame(images[0])
        n_faces = len(result.predictions)
        return f"dry-run ok ({len(images)} photos, {n_faces} faces in first)"

    written = pipeline.enrol_student(account_id=account_id, images=images)
    if not written:
        return "skipped (no face detected in any photo)"
    summary = ", ".join(f"{m}->faceid={fid}" for m, fid in written.items())
    return f"ok ({len(images)} photos; {summary})"


def main() -> int:
    args = _parse_args()
    cfg = AIConfig()
    print(cfg.log_summary())

    pipeline = AttendancePipeline.from_env(cfg)
    repo = pipeline.store_manager.repo

    if args.folder:
        if not os.path.isdir(args.folder):
            print(f"ERROR: {args.folder} is not a directory")
            return 1
        print(f"\n── Bulk enrolment from {args.folder} ──")
        results: dict[str, str] = {}
        for entry in sorted(os.scandir(args.folder), key=lambda e: e.name):
            if not entry.is_dir():
                continue
            sid = entry.name
            image_paths = _collect_images(entry.path)
            results[sid] = _enrol_one(pipeline, repo, sid, image_paths, args.dry_run)
        print()
        for sid, msg in results.items():
            mark = "✓" if msg.startswith("ok") or msg.startswith("dry") else "✗"
            print(f"  {mark}  {sid:<16s}  {msg}")
        return 0

    if args.student:
        if not args.images:
            print("ERROR: --student requires --images <paths...>")
            return 1
        msg = _enrol_one(pipeline, repo, args.student, args.images, args.dry_run)
        mark = "✓" if msg.startswith("ok") or msg.startswith("dry") else "✗"
        print(f"  {mark}  {args.student}  {msg}")
        return 0

    print("ERROR: specify --folder <dir> or --student <id> --images <paths>")
    return 1


if __name__ == "__main__":
    sys.exit(main())
