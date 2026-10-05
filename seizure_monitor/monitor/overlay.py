"""Drawing helpers shared by the live window and the saved event clips."""

import time

import cv2
import numpy as np

from .detector import State
from .pose import SKELETON_CONNECTIONS

# BGR colours
GREEN, ORANGE, RED, GREY, WHITE = (80, 200, 80), (0, 165, 255), (50, 50, 230), (140, 140, 140), (235, 235, 235)
STATE_COLOUR = {State.NORMAL: GREEN, State.RHYTHMIC: ORANGE, State.SEIZURE_EVENT: RED}
PANEL_W = 330
FONT = cv2.FONT_HERSHEY_SIMPLEX


def text(img, s, x, y, colour=WHITE, scale=0.45, thick=1):
    cv2.putText(img, s, (x, y), FONT, scale, colour, thick, cv2.LINE_AA)


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


def draw_clip_banner(frame, out, wall_time: float, source_name: str):
    """Small header for saved clips: when, where, and what the detector thought at that moment."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 26), (0, 0, 0), -1)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(wall_time))
    text(frame, f"{stamp}  {source_name}", 6, 18, WHITE, 0.45)
    label = out.state.value
    (tw, _), _ = cv2.getTextSize(label, FONT, 0.45, 1)
    text(frame, label, w - tw - 8, 18, STATE_COLOUR[out.state], 0.45, 1)


def render_live(frame, result, out, cfg, fps, recording=False, alert=None):
    """Live tuning view: camera frame with skeleton, plus a feature panel on the right."""
    draw_skeleton(frame, result, out)
    h = max(frame.shape[0], 480)
    canvas = np.zeros((h, frame.shape[1] + PANEL_W, 3), np.uint8)
    canvas[:frame.shape[0], :frame.shape[1]] = frame
    if out.state == State.SEIZURE_EVENT and int(time.monotonic() * 3) % 2 == 0:
        cv2.rectangle(canvas, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), RED, 8)
    if recording:
        cv2.circle(canvas, (frame.shape[1] - 50, 22), 7, RED, -1)
        text(canvas, "REC", frame.shape[1] - 38, 27, RED, 0.5, 2)
    _draw_panel(canvas, frame.shape[1], out, cfg, fps)
    if alert:
        _draw_alert_status(canvas, frame.shape[1] + 12, canvas.shape[0] - 60, alert)
    return canvas


def _draw_alert_status(img, x, y, alert):
    if alert["alarm_active"]:
        text(img, "ALARM ON - press S to stop", x, y, RED, 0.5, 2)
    if alert["acknowledged_by"]:
        text(img, f"Acknowledged: {alert['acknowledged_by'][:28]}", x, y - 22, GREEN)
    elif alert["call"]:
        text(img, "ESCALATED - no response, call flag set", x, y - 22, RED)
    elif alert["incident_active"]:
        text(img, "Alert sent - waiting for response", x, y - 22, ORANGE)


def _draw_panel(img, x0, out, cfg, fps):
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
    text(img, f"{fps:4.1f} fps   s: stop alarm   q: quit", x, img.shape[0] - 12, GREY)
    text(img, "amp = RMS movement in shoulder widths", x, img.shape[0] - 32, GREY, 0.38)
