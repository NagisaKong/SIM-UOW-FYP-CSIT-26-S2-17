"""Record labelled behaviour clips for ST-BH-01 / ST-BH-02.

The behaviour service samples at ~1 fps, so this records at 1 fps too — what
you record here is exactly what the detectors see in a live class.

Segment lengths are dictated by the detector defaults (AIConfig), not chosen
arbitrarily:

  * ear_baseline_samples = 20  → the adaptive EAR baseline needs ~20 open-eye
    samples before it will judge anything at all, hence the 25 s baseline.
  * perclos_window_seconds = 60, perclos_threshold = 0.40 → eyes must be shut
    for >40% of a rolling 60 s window, hence 45 s of sustained closure.
  * phone_consec_samples = 5 → a phone episode needs ~5 consecutive samples.

Run:
    python tests/record_behaviour_clips.py --camera 1 --out clips/
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

# label, seconds, on-screen prompt, expect_drowsy, expect_phone
SEGMENTS = [
    ("baseline_open", 25,
     "LOOK AT CAMERA - eyes OPEN, face forward",
     False, False),
    ("drowsy_eyes_closed", 45,
     "CLOSE YOUR EYES and keep them closed",
     True, False),
    ("negative_normal", 30,
     "Normal attentive behaviour - eyes open, no phone",
     False, False),
    ("drowsy_head_down", 20,
     "TILT YOUR HEAD DOWN (chin toward chest)",
     True, False),
    ("phone_use", 20,
     "HOLD A PHONE in front of you and look at it",
     False, True),
    ("negative_no_phone", 20,
     "Hands empty, eyes open - no phone, no drowsiness",
     False, False),
]


@dataclass
class Segment:
    label: str
    frames: list[str]
    expect_drowsy: bool
    expect_phone: bool
    fps: float


_PREVIEW_H = 720  # only the on-screen preview is scaled; frames are saved full-size


def _draw(frame, text: str, remaining: int, seg_i: int, seg_n: int):
    # Downscale for display so a 1080p/4K capture does not open a window
    # larger than the screen. The saved frame is never touched.
    h0, w0 = frame.shape[:2]
    if h0 > _PREVIEW_H:
        out = cv2.resize(frame, (int(w0 * _PREVIEW_H / h0), _PREVIEW_H))
    else:
        out = frame.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w, 90), (0, 0, 0), -1)
    cv2.putText(out, f"[{seg_i}/{seg_n}] {text}", (12, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(out, f"{remaining:>3d}s left   (q = abort)   {w0}x{h0}", (12, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.rectangle(out, (0, h - 6), (w, h), (40, 40, 40), -1)
    return out


def probe_cameras(limit: int = 6) -> list[str]:
    """Enumerate working camera indices and the resolution each can deliver."""
    found = []
    for i in range(limit):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        ok, frame = cap.read()
        if ok:
            h, w = frame.shape[:2]
            found.append(f"--camera {i} ({w}x{h} max)")
        cap.release()
    return found


def open_camera(camera: int, width: int, height: int) -> tuple[cv2.VideoCapture, int, int]:
    """Open the camera and negotiate the capture resolution.

    `cap.set` is a request, not a guarantee — a driver may silently clamp to
    its nearest supported mode. The only reliable resolution is the one read
    back off an actual frame, so that is what gets returned and recorded.

    MJPG is requested first: at 1080p+ many UVC webcams fall back to raw YUY2,
    which exceeds the USB bandwidth budget and collapses the frame rate.
    """
    cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        # Indices are assigned by enumeration order and shift when a device
        # is replugged or the machine reboots, so a hardcoded number goes
        # stale. Report what is actually present instead of just failing.
        raise SystemExit(
            f"Cannot open camera {camera}.\n"
            f"Available now: {probe_cameras() or 'none'}\n"
            f"Re-run with --camera <index>."
        )
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    ok, probe = cap.read()
    if not ok:
        cap.release()
        raise SystemExit(f"Camera {camera} opened but returned no frame")
    got_h, got_w = probe.shape[:2]
    if (got_w, got_h) != (width, height):
        print(f"[warn] requested {width}x{height}, driver gave {got_w}x{got_h}")
    print(f"[record] capture resolution: {got_w}x{got_h}")
    return cap, got_w, got_h


def record(camera: int, out_dir: Path, countdown: int,
           width: int, height: int) -> dict:
    cap, cap_w, cap_h = open_camera(camera, width, height)

    out_dir.mkdir(parents=True, exist_ok=True)
    segments: list[Segment] = []

    try:
        for seg_i, (label, secs, prompt, exp_d, exp_p) in enumerate(SEGMENTS, 1):
            seg_dir = out_dir / label
            seg_dir.mkdir(exist_ok=True)
            for f in seg_dir.glob("*.jpg"):
                f.unlink()

            print(f"\n=== [{seg_i}/{len(SEGMENTS)}] {label} — {secs}s ===")
            print(f"    {prompt}")

            # Get-ready countdown so the subject can settle into position.
            t_end = time.time() + countdown
            while time.time() < t_end:
                ok, frame = cap.read()
                if not ok:
                    continue
                left = int(t_end - time.time()) + 1
                cv2.imshow("record", _draw(frame, f"GET READY: {prompt}",
                                           left, seg_i, len(SEGMENTS)))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise KeyboardInterrupt

            # Capture at exactly 1 fps on a fixed schedule, so a slow read
            # cannot drift the sample spacing the detectors assume.
            frames: list[str] = []
            t0 = time.time()
            for i in range(secs):
                target = t0 + i
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    now = time.time()
                    cv2.imshow("record", _draw(frame, prompt,
                                               secs - i, seg_i, len(SEGMENTS)))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        raise KeyboardInterrupt
                    if now >= target:
                        break
                name = f"{i:04d}.jpg"
                cv2.imwrite(str(seg_dir / name), frame)
                frames.append(f"{label}/{name}")

            segments.append(Segment(label, frames, exp_d, exp_p, 1.0))
            print(f"    captured {len(frames)} frames")
    except KeyboardInterrupt:
        print("\nAborted by user — writing manifest for what was captured.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    manifest = {
        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "camera": camera,
        "capture_width": cap_w,
        "capture_height": cap_h,
        "sampling_fps": 1.0,
        "segments": [asdict(s) for s in segments],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("clips"))
    ap.add_argument("--countdown", type=int, default=5,
                    help="seconds to get into position before each segment")
    # 1080p matches what the production capture client requests (FTD B.1:
    # "Video is requested at 1080p where the device supports it"), and gives
    # the landmarker roughly twice the pixels on a face compared with 720p.
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    args = ap.parse_args()

    m = record(args.camera, args.out, args.countdown, args.width, args.height)
    total = sum(len(s["frames"]) for s in m["segments"])
    print(f"\nWrote {total} frames ({m['capture_width']}x{m['capture_height']}) "
          f"across {len(m['segments'])} segments → {args.out / 'manifest.json'}")
    print("Next: python tests/run_behaviour_st.py --clips", args.out)


if __name__ == "__main__":
    main()
