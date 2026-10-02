"""Seizure monitor entry point.

Runs pose estimation + rhythm detection on a webcam, video file or RTSP stream
and shows a live window with the skeleton, detector state and features, so
thresholds in config.yaml can be tuned by watching.

    python main.py --source 0               # webcam 0
    python main.py --source clip.mp4        # video file (played in real time)
    python main.py --source rtsp://...      # IP camera
    python main.py --source 0 --demo        # shorter onset for live demos
"""

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from monitor.capture import VideoSource
from monitor.detector import Detector, DetectorConfig, State
from monitor.pose import SKELETON_CONNECTIONS, PoseEstimator, ensure_model

BASE_DIR = Path(__file__).resolve().parent
log = logging.getLogger("monitor")

# BGR colours
GREEN, ORANGE, RED, GREY, WHITE = (80, 200, 80), (0, 165, 255), (50, 50, 230), (140, 140, 140), (235, 235, 235)
STATE_COLOUR = {State.NORMAL: GREEN, State.RHYTHMIC: ORANGE, State.SEIZURE_EVENT: RED}
PANEL_W = 330
FONT = cv2.FONT_HERSHEY_SIMPLEX


def main():
    ap = argparse.ArgumentParser(description="Camera-based convulsive seizure monitor")
    ap.add_argument("--source", default="0", help="webcam index, video file path, or RTSP URL")
    ap.add_argument("--config", default=str(BASE_DIR / "config.yaml"))
    ap.add_argument("--demo", action="store_true", help="use demo_onset_s instead of onset_s")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = yaml.safe_load(Path(args.config).read_text())

    det_cfg = DetectorConfig.from_dict(cfg["detector"])
    detector = Detector(det_cfg, demo=args.demo)
    pose = PoseEstimator(cfg["pose"], ensure_model(cfg["pose"], BASE_DIR))
    src = VideoSource(args.source, cfg["capture"]["width"], cfg["capture"]["process_fps"])
    log.info("Source %s (%s), onset %.1f s%s", args.source, "live" if src.is_live else "file",
             detector.onset_s, " [DEMO MODE]" if args.demo else "")

    fps, last_wall = 0.0, None
    play_start = None   # (wall time, video time) for pacing file playback
    try:
        while True:
            item = src.read()
            if item is None:
                log.info("Source ended")
                break
            frame, t = item

            result = pose.process(frame, t)
            out = detector.update(t, result.keypoints)

            if out.event_started:
                log.warning("SEIZURE EVENT started at t=%.1fs (rhythm since t=%.1fs)",
                            out.event_started.detected_at, out.event_started.start)
            if out.event_ended:
                e = out.event_ended
                log.warning("Event ended: %.1fs -> %.1fs, duration %.1fs, confidence %.2f",
                            e.start, e.end, e.duration, e.confidence)

            now = time.monotonic()
            if last_wall is not None:
                fps = 0.9 * fps + 0.1 / max(now - last_wall, 1e-6)   # smoothed for readability
            last_wall = now

            cv2.imshow("Seizure Monitor", render(frame, result, out, det_cfg, fps))

            # Video files: wait so playback runs at real speed (live sources are paced by the camera).
            delay_ms = 1
            if not src.is_live:
                if play_start is None:
                    play_start = (now, t)
                ahead = (t - play_start[1]) - (time.monotonic() - play_start[0])
                delay_ms = max(1, int(ahead * 1000))
            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        src.release()
        pose.close()
        cv2.destroyAllWindows()


# ------------------------------------------------------------------ drawing

def render(frame, result, out, cfg, fps):
    draw_skeleton(frame, result, out)
    h = max(frame.shape[0], 480)
    canvas = np.zeros((h, frame.shape[1] + PANEL_W, 3), np.uint8)
    canvas[:frame.shape[0], :frame.shape[1]] = frame
    if out.state == State.SEIZURE_EVENT and int(time.monotonic() * 3) % 2 == 0:
        cv2.rectangle(canvas, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), RED, 8)
    draw_panel(canvas, frame.shape[1], out, cfg, fps)
    return canvas


def draw_skeleton(frame, result, out):
    if result.all_landmarks is None:
        return
    lms = result.all_landmarks
    for a, b in SKELETON_CONNECTIONS:
        pa, pb = lms[a], lms[b]
        if pa[2] > 0.5 and pb[2] > 0.5:
            cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), GREY, 2)
    # Tracked points: red = rhythmic, green = analysed, grey = not usable in this window.
    for name, (x, y, _) in result.keypoints.items():
        f = out.points.get(name)
        colour = GREY if f is None else (RED if f.rhythmic else GREEN)
        cv2.circle(frame, (int(x), int(y)), 6, colour, -1)


def text(img, s, x, y, colour=WHITE, scale=0.45, thick=1):
    cv2.putText(img, s, (x, y), FONT, scale, colour, thick, cv2.LINE_AA)


def draw_panel(img, x0, out, cfg, fps):
    x, y = x0 + 12, 30
    colour = STATE_COLOUR[out.state]
    cv2.rectangle(img, (x - 4, y - 22), (x0 + PANEL_W - 10, y + 10), colour, -1)
    text(img, out.state.value, x + 4, y + 2, (255, 255, 255), 0.7, 2)

    y += 34
    if not out.body_visible:
        text(img, "No person / shoulders not visible", x, y, ORANGE)
    elif out.warming_up:
        text(img, "Warming up (filling window)...", x, y, ORANGE)
    else:
        text(img, f"Rhythmic limbs: {len(out.rhythmic_limbs)}/{len(out.visible_limbs)} visible"
                  f"  (need {cfg.min_rhythmic_limbs})", x, y)

    # Onset progress: how long rhythm has persisted vs how long it must persist.
    y += 24
    frac = 1.0 if out.state == State.SEIZURE_EVENT else min(out.rhythm_duration_s / out.onset_s, 1.0)
    bar_w = PANEL_W - 30
    cv2.rectangle(img, (x, y - 10), (x + bar_w, y + 4), (60, 60, 60), -1)
    cv2.rectangle(img, (x, y - 10), (x + int(bar_w * frac), y + 4), colour, -1)
    y += 20
    text(img, f"Rhythm {out.rhythm_duration_s:4.1f}s / onset {out.onset_s:.1f}s", x, y)
    y += 20
    conf = out.current_event.confidence if out.current_event else out.window_confidence
    text(img, f"Confidence (heuristic): {conf:.2f}", x, y)

    # Per-limb features: show the limb's strongest point.
    y += 30
    text(img, "limb       band  prom   amp   Hz", x, y, GREY)
    for limb, pts in cfg.limbs.items():
        y += 20
        feats = [out.points[p] for p in pts if p in out.points]
        if not feats:
            text(img, f"{limb:<10} --", x, y, GREY)
            continue
        f = max(feats, key=lambda f: (f.rhythmic, f.prominence))
        prom = ">999" if f.prominence > 999 else f"{f.prominence:4.0f}"
        text(img, f"{limb:<10} {f.band_ratio:4.2f}  {prom}  {f.amplitude:4.2f}  {f.peak_hz:3.1f}",
             x, y, RED if f.rhythmic else WHITE)

    y += 28
    text(img, "thresholds:", x, y, GREY)
    y += 18
    text(img, f"band>={cfg.min_band_ratio}  prom>={cfg.min_prominence}  amp>={cfg.min_amplitude}", x, y, GREY)
    y += 18
    text(img, f"band {cfg.band_hz[0]}-{cfg.band_hz[1]} Hz, window {cfg.window_s}s", x, y, GREY)

    y += 30
    if out.current_event:
        text(img, f"Event running {out.t - out.current_event.start:.0f}s", x, y, RED, 0.5, 1)
    text(img, f"{fps:4.1f} fps   q: quit", x, img.shape[0] - 12, GREY)
    text(img, "amp = RMS movement in shoulder widths", x, img.shape[0] - 32, GREY, 0.38)


if __name__ == "__main__":
    main()
