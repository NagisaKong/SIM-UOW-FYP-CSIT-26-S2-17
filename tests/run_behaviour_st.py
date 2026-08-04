"""Score labelled clips through the REAL behaviour detectors (ST-BH-01/02).

Unlike the unit-level checks in tests/test_system_testing.py, this loads
MediaPipe FaceLandmarker and Ultralytics YOLO for real, replays the clips at
the production 1 fps cadence through the same _EyeClosureModel /
_EpisodeTracker / _phone_owner code the live service uses, and reports
per-segment detection rates against the recorded labels.

Detection (bbox) comes from InsightFace when available, so the face crop fed
to the landmarker is the same crop the live path produces — including the
behaviour_min_face_px cutoff. Pass --no-insightface to let MediaPipe locate
the face in the full frame instead (faster, but skips that cutoff).

Run:
    python tests/run_behaviour_st.py --clips clips/
    python tests/run_behaviour_st.py --clips clips/ --evidence docs/evidence/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.attendancePipeline import AIConfig  # noqa: E402
from core.behaviourAnalysis import (  # noqa: E402
    BehaviourAnalysisService,
    DrowsinessDetector,
    PhoneDetector,
    _EpisodeTracker,
    _EyeClosureModel,
    _HeadPoseModel,
    _StaticBoxFilter,
)


class _Face:
    """Minimal stand-in for Prediction — only bbox/account_id are used."""

    def __init__(self, bbox, account_id=1):
        self.bbox = bbox
        self.account_id = account_id


def _build_detector(use_insightface: bool):
    """Return fn(frame) -> [x1,y1,x2,y2] or None."""
    if not use_insightface:
        return lambda frame: ([0, 0, frame.shape[1], frame.shape[0]], None)
    try:
        from insightface.app import FaceAnalysis

        from core.attendancePipeline import _onnx_providers

        cfg_ctx = int(os.getenv("AI_CTX_ID", "-1"))
        # landmark_3d_68 supplies the head pose the behaviour branch prefers,
        # mirroring what _build_face_analysis loads when AI_BEHAVIOUR is on.
        app = FaceAnalysis(name="buffalo_l",
                           allowed_modules=["detection", "landmark_3d_68"],
                           providers=_onnx_providers(cfg_ctx))
        app.prepare(ctx_id=cfg_ctx, det_size=(640, 640))
        print(f"[st-bh] face detection: InsightFace SCRFD + 3D landmarks "
              f"(ctx_id={cfg_ctx})")

        def detect(frame):
            faces = app.get(frame)
            if not faces:
                return None, None
            f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0])
                    * (f.bbox[3] - f.bbox[1]))
            pose = getattr(f, "pose", None)
            return ([int(v) for v in f.bbox],
                    float(pose[0]) if pose is not None else None)

        return detect
    except Exception as exc:  # noqa: BLE001
        print(f"[st-bh] InsightFace unavailable ({exc}); using full frame")
        return lambda frame: ([0, 0, frame.shape[1], frame.shape[0]], None)


def make_state(cfg):
    """Per-student behaviour state, exactly as the live service holds it.

    In production this lives in `_SessionState.students[account_id]` and
    persists for the whole class. It must therefore be created ONCE and
    shared across every segment: rebuilding it per segment would restart
    the adaptive EAR calibration each time and — worse — let a segment
    where the eyes are shut calibrate its own "open eye" baseline.
    """
    return {
        "eyes": _EyeClosureModel(
            fixed_threshold=cfg.ear_threshold,
            adaptive=cfg.adaptive_ear,
            baseline_samples=cfg.ear_baseline_samples,
            baseline_ratio=cfg.ear_baseline_ratio,
            window_seconds=cfg.perclos_window_seconds,
            perclos_threshold=cfg.perclos_threshold,
        ),
        "head": _HeadPoseModel(
            fixed_threshold=cfg.headpose_pitch_deg,
            adaptive=cfg.adaptive_headpose,
            baseline_samples=cfg.headpose_baseline_samples,
            delta_deg=cfg.headpose_delta_deg,
        ),
        "static_phones": _StaticBoxFilter(
            cfg.phone_static_samples, cfg.phone_static_iou,
        ),
        "drowsy_ep": _EpisodeTracker(cfg.ear_consec_seconds),
        "phone_ep": _EpisodeTracker(float(cfg.phone_consec_samples)),
    }


def run_segment(seg, clips: Path, cfg, drowsy_det, phone_det,
                detect_face, state, t0: float) -> dict:
    """Replay one segment at 1 fps; return per-sample signals + episodes.

    `state` is shared across segments and `t0` continues the session clock,
    so the replay is one continuous lesson rather than six cold starts.
    """
    eyes = state["eyes"]
    head = state["head"]
    static_phones = state["static_phones"]
    drowsy_ep = state["drowsy_ep"]
    phone_ep = state["phone_ep"]

    samples = []
    drowsy_episodes, phone_episodes = [], []
    analysed = 0

    for i, rel in enumerate(seg["frames"]):
        frame = cv2.imread(str(clips / rel))
        if frame is None:
            continue
        now = t0 + i  # production cadence: one sample per second

        bbox, det_pitch = detect_face(frame)
        faces = [_Face(bbox)] if bbox else []

        # ── phone (resolved first; gates the head_pose reason below) ──
        owner_conf = None
        dets = [(b, c) for b, c in phone_det.detect(frame) if c >= cfg.phone_conf]
        static_phones.update([b for b, _ in dets])
        suppressed = sum(1 for b, _ in dets if static_phones.is_static(b))
        for box, conf in dets:
            if static_phones.is_static(box):
                continue  # room fixture, not a phone
            if BehaviourAnalysisService._phone_owner(box, faces) is not None:
                owner_conf = max(owner_conf or 0.0, conf)

        # ── drowsiness ────────────────────────────────────────────────
        reasons: list[str] = []
        obs = None
        if bbox:
            crop = BehaviourAnalysisService._face_crop(frame, bbox)
            too_small = min(crop.shape[:2]) < cfg.behaviour_min_face_px
            obs = None if too_small else drowsy_det.observe(crop)
        # Detector pose first — it survives the head-down frames on which the
        # landmarker returns nothing, which is exactly where it is needed.
        # Both sources are normalised to "degrees downward" (see the service).
        if det_pitch is not None:
            down_deg = -det_pitch
        elif obs is not None and obs["pitch"] is not None:
            down_deg = abs(obs["pitch"])
        else:
            down_deg = None
        if obs is not None or down_deg is not None:
            analysed += 1
        # Gated by phone use, matching BehaviourAnalysisService: a student
        # bent over a phone is not also reported as drowsy.
        if down_deg is not None and head.observe(down_deg) and owner_conf is None:
            reasons.append("head_pose")
        if obs is not None:
            if eyes.observe(obs["ear"], now):
                reasons.append("eyes_closed")
            if obs["mar"] > cfg.mar_threshold:
                reasons.append("yawn")
        ep = drowsy_ep.update(bool(reasons), now,
                              {"reasons": reasons} if reasons else None)
        if ep:
            drowsy_episodes.append(ep)

        # ── phone episode bookkeeping (detection ran above) ───────────
        ep = phone_ep.update(owner_conf is not None, now,
                             {"conf": owner_conf} if owner_conf else None)
        if ep:
            phone_episodes.append(ep)

        samples.append({
            "i": i,
            "face": bool(bbox),
            "ear": obs["ear"] if obs else None,
            "mar": obs["mar"] if obs else None,
            "down_deg": round(down_deg, 1) if down_deg is not None else None,
            "pitch_source": ("detector" if det_pitch is not None
                             else ("landmarker" if obs else None)),
            "ear_threshold": round(eyes.threshold, 3) if eyes.baseline else None,
            "pitch_threshold": round(head.threshold, 1) if head.baseline else None,
            "perclos": round(eyes.perclos(), 2),
            "calibrating": eyes.calibrating() or head.calibrating(),
            "static_suppressed": suppressed,
            "drowsy_reasons": reasons,
            "drowsy_confirmed": drowsy_ep.is_confirmed(now),
            "phone_conf": owner_conf,
            "phone_confirmed": phone_ep.is_confirmed(now),
        })

    n = len(samples)
    # Drowsiness recall is measured over samples the detector could actually
    # judge: the adaptive baseline deliberately withholds a verdict while
    # calibrating, and counting those as misses would understate a detector
    # that is working exactly as designed.
    judgeable = [s for s in samples if not s["calibrating"] and s["face"]]
    # Phone detection has NO calibration phase — it is YOLO plus a geometric
    # attribution to a detected face. Its denominator is therefore every
    # sample with a face, not the EAR-judgeable subset.
    phone_scope = [s for s in samples if s["face"]]
    drowsy_hits = sum(1 for s in judgeable if s["drowsy_reasons"])
    phone_hits = sum(1 for s in phone_scope if s["phone_conf"] is not None)

    return {
        "label": seg["label"],
        "expect_drowsy": seg["expect_drowsy"],
        "expect_phone": seg["expect_phone"],
        "samples": n,
        "faces_found": sum(1 for s in samples if s["face"]),
        "analysed": analysed,
        "judgeable": len(judgeable),
        "drowsy_sample_rate": round(drowsy_hits / len(judgeable), 3) if judgeable else None,
        "phone_sample_rate": round(phone_hits / len(phone_scope), 3) if phone_scope else None,
        "drowsy_episodes": len(drowsy_episodes),
        "phone_episodes": len(phone_episodes),
        "drowsy_episode_seconds": sum(e["duration"] for e in drowsy_episodes),
        "phone_episode_seconds": sum(e["duration"] for e in phone_episodes),
        "_samples": samples,
    }


def verdicts(results: list[dict]) -> dict:
    """Fold per-segment numbers into the two ST-BH pass criteria."""
    pos_d = [r for r in results if r["expect_drowsy"]]
    neg_d = [r for r in results if not r["expect_drowsy"]]
    pos_p = [r for r in results if r["expect_phone"]]
    neg_p = [r for r in results if not r["expect_phone"]]

    def _rate(rows, key):
        vals = [r[key] for r in rows if r[key] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    d_recall = _rate(pos_d, "drowsy_sample_rate")
    d_fa = _rate(neg_d, "drowsy_sample_rate")
    p_recall = _rate(pos_p, "phone_sample_rate")
    p_fa = _rate(neg_p, "phone_sample_rate")

    return {
        "ST-BH-01": {
            "criterion": "drowsiness recall >= 0.80 on labelled drowsy segments",
            "recall": d_recall,
            "false_alarm_rate_on_negative": d_fa,
            "episodes_confirmed": sum(r["drowsy_episodes"] for r in pos_d),
            "pass": bool(d_recall is not None and d_recall >= 0.80),
        },
        "ST-BH-02": {
            "criterion": "phone-use recall >= 0.75 on labelled phone segments",
            "recall": p_recall,
            "false_alarm_rate_on_negative": p_fa,
            "episodes_confirmed": sum(r["phone_episodes"] for r in pos_p),
            "pass": bool(p_recall is not None and p_recall >= 0.75),
        },
    }


def write_evidence(path: Path, cfg, results: list[dict], v: dict,
                   manifest: dict, elapsed: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    L = []
    L.append("# ST-BH-01 / ST-BH-02 — measured detection rates\n")
    L.append(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by "
             "`tests/run_behaviour_st.py`.\n")
    L.append("Real MediaPipe FaceLandmarker + Ultralytics YOLO inference over "
             "labelled clips recorded at the production 1 fps cadence, replayed "
             "through the same `_EyeClosureModel` / `_EpisodeTracker` / "
             "`_phone_owner` code as the live service.\n")
    L.append(f"- Clips recorded: {manifest.get('recorded_at', 'n/a')}")
    L.append(f"- Total frames scored: {sum(r['samples'] for r in results)}")
    L.append(f"- Wall-clock: {elapsed:.1f}s\n")

    L.append("## Detector configuration\n")
    L.append("| Parameter | Value |")
    L.append("|---|---|")
    for k in ("ear_threshold", "adaptive_ear", "ear_baseline_samples",
              "ear_baseline_ratio", "perclos_window_seconds",
              "perclos_threshold", "ear_consec_seconds", "mar_threshold",
              "headpose_pitch_deg", "phone_conf", "phone_consec_samples",
              "phone_model", "phone_imgsz", "behaviour_min_face_px"):
        L.append(f"| `{k}` | {getattr(cfg, k)} |")
    L.append("")

    L.append("## Per-segment results\n")
    L.append("| Segment | Label | Samples | Faces | Analysed | Drowsy rate | "
             "Phone rate | Drowsy eps | Phone eps |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        lab = []
        if r["expect_drowsy"]:
            lab.append("drowsy")
        if r["expect_phone"]:
            lab.append("phone")
        L.append(
            f"| `{r['label']}` | {'+'.join(lab) or 'negative'} | {r['samples']} | "
            f"{r['faces_found']} | {r['analysed']} | "
            f"{r['drowsy_sample_rate']} | {r['phone_sample_rate']} | "
            f"{r['drowsy_episodes']} | {r['phone_episodes']} |"
        )
    L.append("")

    L.append("## Verdicts\n")
    L.append("| ID | Criterion | Measured recall | False alarms on negatives | "
             "Episodes | Pass? |")
    L.append("|---|---|---|---|---|---|")
    for tid, d in v.items():
        L.append(
            f"| {tid} | {d['criterion']} | **{d['recall']}** | "
            f"{d['false_alarm_rate_on_negative']} | {d['episodes_confirmed']} | "
            f"**{'Y' if d['pass'] else 'N'}** |"
        )
    L.append("")
    L.append("> Recall is computed over samples the detector could actually "
             "judge — frames where a face was found and the adaptive EAR "
             "baseline had finished calibrating. Calibration samples are "
             "excluded because the model deliberately withholds a verdict "
             "during that window.")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n[st-bh] evidence written → {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips", type=Path, default=Path("clips"))
    ap.add_argument("--evidence", type=Path, default=None,
                    help="directory to write the evidence markdown into")
    ap.add_argument("--no-insightface", action="store_true")
    ap.add_argument("--json", type=Path, default=None,
                    help="dump full per-sample signals for inspection")
    args = ap.parse_args()

    manifest_path = args.clips / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(
            f"No manifest at {manifest_path}. Record clips first:\n"
            "    python tests/record_behaviour_clips.py --camera 1"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    cfg = AIConfig()
    print(f"[st-bh] ear={cfg.ear_threshold} adaptive={cfg.adaptive_ear} "
          f"perclos={cfg.perclos_threshold}@{cfg.perclos_window_seconds}s "
          f"phone_conf={cfg.phone_conf} model={cfg.phone_model}")

    drowsy_det = DrowsinessDetector()
    phone_det = PhoneDetector(cfg.phone_conf, cfg.phone_model, cfg.phone_imgsz,
                              want_gpu=cfg.ctx_id >= 0)
    detect_face = _build_detector(not args.no_insightface)

    # One shared state and one continuous clock across all segments — the
    # replay models a single lesson, which is the only way the adaptive EAR
    # baseline established during `baseline_open` carries into the segments
    # that depend on it.
    state = make_state(cfg)
    session_clock = time.time()
    t0 = time.time()
    results = []
    for seg in manifest["segments"]:
        print(f"[st-bh] scoring {seg['label']} ({len(seg['frames'])} frames)…")
        results.append(run_segment(seg, args.clips, cfg, drowsy_det,
                                   phone_det, detect_face, state,
                                   session_clock))
        session_clock += len(seg["frames"])
    elapsed = time.time() - t0

    v = verdicts(results)
    print("\n" + "=" * 62)
    for tid, d in v.items():
        print(f"{tid}: recall={d['recall']} "
              f"false_alarm={d['false_alarm_rate_on_negative']} "
              f"episodes={d['episodes_confirmed']} "
              f"→ {'PASS' if d['pass'] else 'FAIL'}")
    print("=" * 62)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            {"results": results, "verdicts": v}, indent=2, default=str),
            encoding="utf-8")
        print(f"[st-bh] per-sample signals → {args.json}")

    if args.evidence:
        stamp = time.strftime("%Y%m%d_%H%M")
        write_evidence(args.evidence / f"behaviour_st_{stamp}.md",
                       cfg, results, v, manifest, elapsed)


if __name__ == "__main__":
    main()
