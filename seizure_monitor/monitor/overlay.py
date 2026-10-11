"""Drawing helpers shared by the live window and the saved event clips."""

import time

import cv2
import numpy as np

from .detector import State
from .motion import bed_pixels
from .pose import SKELETON_CONNECTIONS

# BGR colours
GREEN, ORANGE, RED, GREY, WHITE = (80, 200, 80), (0, 165, 255), (50, 50, 230), (140, 140, 140), (235, 235, 235)
STATE_COLOUR = {State.NORMAL: GREEN, State.RHYTHMIC: ORANGE, State.MOVEMENT: ORANGE, State.SEIZURE_EVENT: RED}
CYAN = (200, 180, 40)
PANEL_W = 330
PANEL_MIN_H = 600
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


def draw_bed_area(frame, bed_area, active=False):
    """Outline of the region detector A watches (only movement inside it counts)."""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = bed_pixels(bed_area, w, h)
    cv2.rectangle(frame, (x0 + 1, y0 + 1), (x1 - 2, y1 - 2), ORANGE if active else CYAN, 2)
    text(frame, "bed area", x0 + 6, y1 - 8, ORANGE if active else CYAN, 0.45, 1)


def draw_clip_banner(frame, out, wall_time: float, source_name: str):
    """Small header for saved clips: when, where, and what the detector thought at that moment."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 26), (0, 0, 0), -1)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(wall_time))
    text(frame, f"{stamp}  {source_name}", 6, 18, WHITE, 0.45)
    label = out.state.value
    (tw, _), _ = cv2.getTextSize(label, FONT, 0.45, 1)
    text(frame, label, w - tw - 8, 18, STATE_COLOUR[out.state], 0.45, 1)


def draw_paused_banner(frame, wall_time: float, source_name: str):
    """Grey banner for the web stream while monitoring is paused."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 26), (60, 60, 60), -1)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(wall_time))
    text(frame, f"{stamp}  {source_name}", 6, 18, WHITE, 0.45)
    (tw, _), _ = cv2.getTextSize("PAUSED", FONT, 0.45, 1)
    text(frame, "PAUSED", w - tw - 8, 18, ORANGE, 0.45, 1)


def render_paused(frame, fps, alert=None):
    """Live window while monitoring is paused: dimmed video, no detection."""
    h = max(frame.shape[0], PANEL_MIN_H)
    canvas = np.zeros((h, frame.shape[1] + PANEL_W, 3), np.uint8)
    canvas[:frame.shape[0], :frame.shape[1]] = (frame * 0.55).astype(np.uint8)
    x0 = frame.shape[1]
    x, y = x0 + 12, 30
    cv2.rectangle(canvas, (x - 4, y - 22), (x0 + PANEL_W - 10, y + 10), GREY, -1)
    text(canvas, "PAUSED", x + 4, y + 2, (255, 255, 255), 0.7, 2)
    for i, line in enumerate(["Monitoring is paused:", "no detection, alarms or recording.", "",
                              "Press P (or the dashboard button)", "to resume when the person is in bed."]):
        text(canvas, line, x, y + 40 + 22 * i)
    if alert:
        _draw_alert_status(canvas, x, canvas.shape[0] - 60, alert)
    text(canvas, f"{fps:4.1f} fps   p: resume   q: quit", x, canvas.shape[0] - 12, GREY)
    return canvas


def render_live(frame, result, out, cfg, mcfg, fps, recording=False, alert=None):
    """Live tuning view: camera frame with skeleton and bed area, plus a feature panel on the right.

    out is the fused output (monitor/fusion.py); cfg/mcfg are the rhythm and motion detector configs.
    """
    draw_skeleton(frame, result, out)
    if out.motion is not None:
        draw_bed_area(frame, mcfg.bed_area, out.motion.above)
    h = max(frame.shape[0], PANEL_MIN_H)
    canvas = np.zeros((h, frame.shape[1] + PANEL_W, 3), np.uint8)
    canvas[:frame.shape[0], :frame.shape[1]] = frame
    if out.state == State.SEIZURE_EVENT and int(time.monotonic() * 3) % 2 == 0:
        cv2.rectangle(canvas, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), RED, 8)
    if recording:
        cv2.circle(canvas, (frame.shape[1] - 50, 22), 7, RED, -1)
        text(canvas, "REC", frame.shape[1] - 38, 27, RED, 0.5, 2)
    _draw_panel(canvas, frame.shape[1], out, cfg, mcfg, fps)
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


def _draw_panel(img, x0, fused, cfg, mcfg, fps):
    x, y = x0 + 12, 30
    colour = STATE_COLOUR[fused.state]
    cv2.rectangle(img, (x - 4, y - 22), (x0 + PANEL_W - 10, y + 10), colour, -1)
    text(img, fused.state.value, x + 4, y + 2, (255, 255, 255), 0.7, 2)
    bar_w = PANEL_W - 30

    # --- Detector A: movement in the bed area
    y += 32
    text(img, "A  Movement in bed area", x, y, CYAN, 0.5, 1)
    m = fused.motion
    if m is None:
        y += 20
        text(img, "off (motion.enabled: false)", x, y, GREY)
    else:
        y += 16
        frac = min(m.intensity / (2 * mcfg.min_speed), 1.0)        # threshold sits at the middle of the bar
        cv2.rectangle(img, (x, y), (x + bar_w, y + 12), (60, 60, 60), -1)
        cv2.rectangle(img, (x, y), (x + int(bar_w * frac), y + 12), RED if m.above else CYAN, -1)
        cv2.line(img, (x + bar_w // 2, y - 3), (x + bar_w // 2, y + 15), WHITE, 1)
        y += 30
        text(img, f"speed {m.intensity:.3f} (thr {mcfg.min_speed})  sustained {m.active_fraction:4.0%}", x, y)
        if mcfg.bed_area is None:
            y += 18
            text(img, "bed area not set: whole frame counts", x, y, ORANGE, 0.4)

    # --- Detector B: rhythmic limbs (pose)
    out = fused.pose
    y += 30
    text(img, "B  Rhythmic limbs (pose)", x, y, CYAN, 0.5, 1)
    y += 22
    if not out.body_visible:
        text(img, "No person / shoulders not visible", x, y, ORANGE)
    elif out.warming_up:
        text(img, "Warming up (filling window)...", x, y, ORANGE)
    else:
        text(img, f"Rhythmic limbs: {len(out.rhythmic_limbs)}/{len(out.visible_limbs)} visible"
                  f"  (need {cfg.min_rhythmic_limbs})", x, y)

    # Onset progress: how long rhythm has persisted vs how long it must persist.
    y += 16
    frac = 1.0 if out.current_event else min(out.rhythm_duration_s / out.onset_s, 1.0)
    cv2.rectangle(img, (x, y), (x + bar_w, y + 12), (60, 60, 60), -1)
    cv2.rectangle(img, (x, y), (x + int(bar_w * frac), y + 12), ORANGE if frac < 1 else RED, -1)
    y += 30
    text(img, f"Rhythm {out.rhythm_duration_s:4.1f}s / onset {out.onset_s:.1f}s", x, y)

    # Per-limb features: show the limb's strongest point.
    y += 24
    text(img, "limb       band  prom   amp   Hz", x, y, GREY)
    for limb, pts in cfg.limbs.items():
        y += 19
        feats = [out.points[p] for p in pts if p in out.points]
        if not feats:
            text(img, f"{limb:<10} --", x, y, GREY)
            continue
        f = max(feats, key=lambda f: (f.rhythmic, f.prominence))
        prom = ">999" if f.prominence > 999 else f"{f.prominence:4.0f}"
        text(img, f"{limb:<10} {f.band_ratio:4.2f}  {prom}  {f.amplitude:4.2f}  {f.peak_hz:3.1f}",
             x, y, RED if f.rhythmic else WHITE)
    y += 18
    text(img, f"thr: band>={cfg.min_band_ratio} prom>={cfg.min_prominence} amp>={cfg.min_amplitude}", x, y, GREY, 0.4)

    # --- Event
    y += 30
    if fused.current_event:
        e = fused.current_event
        text(img, f"Event running {fused.t - e.start:.0f}s  ({e.trigger})", x, y, RED, 0.5, 1)
        y += 20
        text(img, f"Confidence (heuristic): {e.confidence:.2f}", x, y)
    text(img, f"{fps:4.1f} fps   s: stop alarm   p: pause   q: quit", x, img.shape[0] - 12, GREY)
