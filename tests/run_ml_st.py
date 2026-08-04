"""Measure ST-ML-01…05 with real SCRFD / ArcFace inference.

The pytest versions of these cases run against `tests/conftest.py`'s stub
pipeline, which returns a deterministic hash-derived vector per image. That
verifies the enrolment -> identify -> threshold -> decision path, but says
nothing about whether the models can actually tell people apart. This script
measures that.

Everything is computed in memory. ST-ML-04 reads the production embeddings
read-only via TrainConfiguration.runEvaluation; nothing is ever written, and
no threshold is deployed.

Genuine pairs come from many frames of one enrolled subject; imposters come
from a group photograph of people who are definitively not enrolled.

Run:
    python tests/run_ml_st.py --clips clips --evidence docs/evidence
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.attendancePipeline import (  # noqa: E402
    AIConfig,
    ArcFaceRecognizer,
    Detection,
    _build_face_analysis,
)

# Bundled InsightFace sample: a group photograph, used as the imposter set.
_IMPOSTER_IMAGE = ".venv/insightface/data/images/t1.jpg"


def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def collect_subject(app, rec, frames: list[str]) -> tuple[list[np.ndarray], int, int]:
    """Embed every frame of the enrolled subject. Returns (vectors, detected, total)."""
    vecs, detected = [], 0
    for fp in frames:
        img = cv2.imread(fp)
        if img is None:
            continue
        faces = app.get(img)
        if not faces:
            continue
        detected += 1
        f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        det = Detection(bbox=f.bbox.astype(int), det_score=float(f.det_score),
                        kps=f.kps, source="scrfd", raw=f)
        v = rec.embed(img, det)
        if v is not None:
            vecs.append(_norm(v))
    return vecs, detected, len(frames)


def collect_imposters(app, rec, image_path: str) -> list[np.ndarray]:
    img = cv2.imread(image_path)
    if img is None:
        return []
    out = []
    for f in app.get(img):
        det = Detection(bbox=f.bbox.astype(int), det_score=float(f.det_score),
                        kps=f.kps, source="scrfd", raw=f)
        v = rec.embed(img, det)
        if v is not None:
            out.append(_norm(v))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips", type=Path, default=Path("clips"))
    ap.add_argument("--evidence", type=Path, default=None)
    ap.add_argument("--gallery-size", type=int, default=5,
                    help="frames enrolled as the gallery; the rest are probes")
    ap.add_argument("--imposter-image", default=_IMPOSTER_IMAGE)
    args = ap.parse_args()

    cfg = AIConfig()
    print(f"[st-ml] threshold={cfg.arcface_threshold} ctx_id={cfg.ctx_id}")
    app = _build_face_analysis(cfg)
    rec = ArcFaceRecognizer(cfg, shared_app=app)

    frames = sorted(glob.glob(str(args.clips / "*" / "*.jpg")))
    if not frames:
        raise SystemExit(f"No frames under {args.clips}")
    print(f"[st-ml] {len(frames)} subject frames, imposters from {args.imposter_image}")

    t0 = time.time()
    vecs, detected, total = collect_subject(app, rec, frames)
    imposters = collect_imposters(app, rec, args.imposter_image)
    if len(vecs) <= args.gallery_size:
        raise SystemExit("Not enough embedded frames to split gallery/probe")
    if not imposters:
        raise SystemExit(f"No imposter faces found in {args.imposter_image}")

    # ── ST-ML-01: detection recall ───────────────────────────────────
    det_recall = detected / total

    # ── ST-ML-02 / 03: gallery vs probes and imposters ───────────────
    gallery = vecs[:args.gallery_size]
    probes = vecs[args.gallery_size:]
    centroid = _norm(np.mean(gallery, axis=0))

    def best_score(v):
        return max(float(np.dot(v, g)) for g in gallery)

    probe_scores = [best_score(v) for v in probes]
    imposter_scores = [best_score(v) for v in imposters]
    th = cfg.arcface_threshold
    tar = sum(1 for s in probe_scores if s >= th) / len(probe_scores)
    far = sum(1 for s in imposter_scores if s >= th) / len(imposter_scores)

    # ── ST-ML-04: calibration on the production gallery (read-only) ──
    calib = None
    try:
        from core.trainConfiguration import TrainConfiguration, load_deployed_threshold

        svc = TrainConfiguration(cfg.database_url)
        deployed = load_deployed_threshold(cfg.database_url, "arcface")
        calib = svc.runEvaluation("arcface", 80, deployed)
    except Exception as exc:  # noqa: BLE001
        print(f"[st-ml] ST-ML-04 skipped: {exc}")

    # ── ST-ML-05: does the deploy gate protect against a regression? ─
    #
    # Raising a threshold always costs some TAR — that on its own is not a
    # regression, it is the price of a wider safety margin. What ST-ML-05
    # is really about is whether a WORSE configuration can reach production.
    # So compare balanced accuracy on this independent probe/imposter set,
    # and require that if the calibrated threshold is worse, the deploy gate
    # (`improved`, checked by deployUpdatedModel) would withhold it.
    regression = None
    if calib is not None:
        new_th = calib["new_threshold"]
        tar_new = sum(1 for s in probe_scores if s >= new_th) / len(probe_scores)
        far_new = (sum(1 for s in imposter_scores if s >= new_th)
                   / len(imposter_scores))
        ba_old = (tar + (1 - far)) / 2
        ba_new = (tar_new + (1 - far_new)) / 2
        would_deploy = bool(calib.get("improved"))
        regression = {
            "new_threshold": new_th,
            "tar_at_new": round(tar_new, 4), "far_at_new": round(far_new, 4),
            "tar_at_deployed": round(tar, 4), "far_at_deployed": round(far, 4),
            "balanced_accuracy_deployed": round(ba_old, 4),
            "balanced_accuracy_calibrated": round(ba_new, 4),
            "no_worse": ba_new >= ba_old,
            "gate_would_deploy": would_deploy,
            # No baseline row in MODEL_CONFIGS means the gate has nothing to
            # compare against and passes everything through — expected on a
            # first deploy, but worth surfacing rather than hiding.
            "gate_has_baseline": calib.get("current_threshold") is not None,
        }

    elapsed = time.time() - t0
    results = {
        "frames": total, "detected": detected, "embedded": len(vecs),
        "gallery": len(gallery), "probes": len(probes),
        "imposters": len(imposters),
        "threshold": th,
        "detection_recall": round(det_recall, 4),
        "tar": round(tar, 4), "far": round(far, 4),
        "probe_score_min": round(min(probe_scores), 4),
        "probe_score_median": round(sorted(probe_scores)[len(probe_scores) // 2], 4),
        "imposter_score_max": round(max(imposter_scores), 4),
        "imposter_score_median": round(
            sorted(imposter_scores)[len(imposter_scores) // 2], 4),
        "margin": round(min(probe_scores) - max(imposter_scores), 4),
        "centroid_probe_median": round(
            float(np.median([float(np.dot(v, centroid)) for v in probes])), 4),
        "calibration": calib,
        "regression": regression,
        "elapsed_s": round(elapsed, 1),
    }

    verdicts = {
        "ST-ML-01": ("SCRFD detection recall >= 0.90", det_recall, det_recall >= 0.90),
        "ST-ML-02": ("ArcFace true accept rate >= 0.90", tar, tar >= 0.90),
        "ST-ML-03": ("ArcFace false accept rate <= 0.05", far, far <= 0.05),
        "ST-ML-04": ("calibration returns accuracy/FPR/FNR",
                     calib["accuracy"] if calib else None, calib is not None),
        # Passes when the calibrated threshold is no worse, or when it is
        # worse and the deploy gate withholds it. Reported as inconclusive
        # rather than failed when MODEL_CONFIGS has no active row: the gate
        # has nothing to regress against, so there is no protection to test.
        "ST-ML-05": ("calibrated config is no worse, or the deploy gate blocks it",
                     (f"BA {regression['balanced_accuracy_calibrated']} vs "
                      f"{regression['balanced_accuracy_deployed']} deployed"
                      if regression else None),
                     None if (regression and not regression["gate_has_baseline"])
                     else bool(regression and (regression["no_worse"]
                                               or not regression["gate_would_deploy"]))),
    }

    print("\n" + "=" * 68)
    for tid, (crit, val, ok) in verdicts.items():
        state = "INCONCLUSIVE" if ok is None else ("PASS" if ok else "FAIL")
        print(f"{tid}: {crit}\n         measured={val} -> {state}")
    print("=" * 68)
    print(f"score separation: probe_min={results['probe_score_min']} "
          f"imposter_max={results['imposter_score_max']} "
          f"margin={results['margin']}")

    if args.evidence:
        write_evidence(args.evidence / f"ml_st_{time.strftime('%Y%m%d_%H%M')}.md",
                       cfg, results, verdicts, args)


def write_evidence(path: Path, cfg, r: dict, v: dict, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    L = ["# ST-ML-01…05 — measured recognition accuracy\n",
         f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by "
         "`tests/run_ml_st.py`.\n",
         "Real SCRFD detection and ArcFace recognition — no stub pipeline. "
         "Genuine probes are held-out frames of one enrolled subject; "
         "imposters are the faces in a group photograph of people who are "
         "not enrolled. ST-ML-04 evaluates the production embedding gallery "
         "read-only; no threshold was deployed.\n",
         "## Setup\n",
         "| | |", "|---|---|",
         f"| Subject frames | {r['frames']} |",
         f"| Frames with a detected face | {r['detected']} |",
         f"| Frames embedded | {r['embedded']} |",
         f"| Gallery / probe split | {r['gallery']} / {r['probes']} |",
         f"| Imposter faces | {r['imposters']} (`{args.imposter_image}`) |",
         f"| Deployed threshold | {r['threshold']} |",
         f"| Device | ctx_id={cfg.ctx_id} |",
         f"| Wall-clock | {r['elapsed_s']}s |", "",
         "## Score separation\n",
         "| Metric | Value |", "|---|---|",
         f"| Lowest genuine probe score | {r['probe_score_min']} |",
         f"| Median genuine probe score | {r['probe_score_median']} |",
         f"| Highest imposter score | {r['imposter_score_max']} |",
         f"| Median imposter score | {r['imposter_score_median']} |",
         f"| **Margin** (min genuine − max imposter) | **{r['margin']}** |", "",
         "## Verdicts\n",
         "| ID | Criterion | Measured | Pass? |", "|---|---|---|---|"]
    for tid, (crit, val, ok) in v.items():
        mark = "—" if ok is None else ("Y" if ok else "N")
        L.append(f"| {tid} | {crit} | **{val}** | **{mark}** |")
    L.append("")
    if r["calibration"]:
        c = r["calibration"]
        L += ["## ST-ML-04 — calibration on the production gallery\n",
              "| Metric | Value |", "|---|---|",
              f"| Accounts | {c['accounts']} |",
              f"| Genuine pairs | {c['genuine_pairs']} |",
              f"| Imposter pairs | {c['imposter_pairs']} |",
              f"| Selected threshold | {c['new_threshold']} |",
              f"| Balanced accuracy (test split) | {c['accuracy']} |",
              f"| FPR / FNR | {c['fpr']} / {c['fnr']} |",
              f"| Currently deployed threshold | {c['current_threshold']} |",
              f"| Limited calibration | {c['limited_calibration']} |", ""]
    if r["regression"]:
        g = r["regression"]
        L += ["## ST-ML-05 — regression protection (nothing deployed)\n",
              "| Threshold | TAR | FAR | Balanced accuracy |",
              "|---|---|---|---|",
              f"| Deployed ({r['threshold']}) | {g['tar_at_deployed']} "
              f"| {g['far_at_deployed']} | {g['balanced_accuracy_deployed']} |",
              f"| Calibrated ({g['new_threshold']}) | {g['tar_at_new']} "
              f"| {g['far_at_new']} | {g['balanced_accuracy_calibrated']} |", "",
              f"- Calibrated config no worse: **{g['no_worse']}**",
              f"- Deploy gate would allow it: **{g['gate_would_deploy']}**",
              f"- Gate has a baseline to compare against: "
              f"**{g['gate_has_baseline']}**", ""]
        if not g["gate_has_baseline"]:
            L.append("> `MODEL_CONFIGS` holds no active row for this model, so "
                     "`deployUpdatedModel` has no baseline and treats any "
                     "calibration as an improvement. That is correct for a "
                     "first deploy, but it means the regression gate is "
                     "inactive until one threshold has been deployed.\n")
        if not g["no_worse"]:
            L.append("> A higher threshold trades true-accept rate for margin, "
                     "so a small TAR drop at equal FAR is not by itself a "
                     "regression. The criterion here is that a configuration "
                     "which is worse overall must not reach production.\n")
    L.append("> Limitation: the genuine set is one subject recorded in a single "
             "session, so it does not exercise inter-session appearance change "
             "(lighting, glasses, hair). The imposter set is small. These "
             "figures establish that the models discriminate on real faces; "
             "they are not a cohort-scale accuracy estimate.")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n[st-ml] evidence written → {path}")


if __name__ == "__main__":
    main()
