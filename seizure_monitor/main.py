"""Seizure monitor entry point.

Runs pose estimation + rhythm detection on a webcam, video file or RTSP/HTTP
stream and shows a live window with the skeleton, detector state and features,
so thresholds in config.yaml can be tuned by watching. Detected events are
saved as video clips and logged in an SQLite database, and trigger a local
alarm plus Telegram alerts (secrets come from .env, never from code). A web
dashboard (live view, event review, CSV diary) runs at http://127.0.0.1:8000.

    python main.py --source 0               # webcam 0
    python main.py --source clip.mp4        # video file (played in real time)
    python main.py --source rtsp://...      # IP camera
    python main.py --source 0 --demo        # shorter onset for live demos

Keys in the live window: s = stop alarm (counts as "someone is here"), q = quit.
"""

import argparse
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import yaml
from dotenv import load_dotenv

from dashboard.app import DashboardServer, LiveState, create_app
from monitor.alerts import AlertManager, LocalAlarm, TelegramClient
from monitor.capture import VideoSource
from monitor.detector import Detector, DetectorConfig
from monitor.overlay import draw_clip_banner, draw_skeleton, render_live
from monitor.pose import PoseEstimator, ensure_model
from monitor.recorder import EventRecorder, PrivacyFilter
from monitor.storage import EventStore

BASE_DIR = Path(__file__).resolve().parent
log = logging.getLogger("monitor")


def iso(wall_time: float) -> str:
    return datetime.fromtimestamp(wall_time).isoformat(timespec="seconds")


def build_alerts(cfg: dict, source_name: str) -> AlertManager:
    acfg = cfg["alerts"]
    telegram = None
    if acfg["telegram"]["enabled"]:
        token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
        if token and chat:
            telegram = TelegramClient(token, chat, acfg["telegram"])
            telegram.start()
            log.info("Telegram alerts enabled")
        else:
            log.warning("Telegram enabled in config but TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing in .env: "
                        "only the local alarm will sound")
    return AlertManager(acfg, source_name, LocalAlarm(acfg["alarm"]), telegram)


def main():
    ap = argparse.ArgumentParser(description="Camera-based convulsive seizure monitor")
    ap.add_argument("--source", default="0", help="webcam index, video file path, or RTSP/HTTP URL")
    ap.add_argument("--config", default=str(BASE_DIR / "config.yaml"))
    ap.add_argument("--demo", action="store_true", help="use demo_onset_s instead of onset_s")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_dotenv(BASE_DIR / ".env")
    cfg = yaml.safe_load(Path(args.config).read_text())
    source_name = cfg["monitor"]["source_name"]

    det_cfg = DetectorConfig.from_dict(cfg["detector"])
    detector = Detector(det_cfg, demo=args.demo)
    pose = PoseEstimator(cfg["pose"], ensure_model(cfg["pose"], BASE_DIR))
    src = VideoSource(args.source, cfg["capture"]["width"], cfg["capture"]["process_fps"])
    log.info("Source %s (%s), onset %.1f s%s", args.source, "live" if src.is_live else "file",
             detector.onset_s, " [DEMO MODE]" if args.demo else "")

    data_dir = BASE_DIR / cfg["recording"]["data_dir"]
    store = EventStore(data_dir / "events.db")
    recorder = None
    if cfg["recording"]["enabled"]:
        recorder = EventRecorder(cfg["recording"], cfg["privacy"], data_dir / "clips", source_name)
    snapshot_privacy = PrivacyFilter(cfg["privacy"])   # only used for snapshots when recording is off
    alerts = build_alerts(cfg, source_name)

    live, server = None, None
    dcfg = cfg["dashboard"]
    if dcfg["enabled"]:
        live = LiveState(dcfg["stream_jpeg_quality"], alert_status=alerts.status)
        stream_privacy = PrivacyFilter(cfg["privacy"]) if dcfg["blur_stream"] else None
        server = DashboardServer(create_app(store, data_dir, live, dcfg["stream_fps"]), dcfg["host"], dcfg["port"])
        server.start()
        log.info("Dashboard: http://%s:%d", dcfg["host"], dcfg["port"])
        if dcfg["host"] not in ("127.0.0.1", "localhost"):
            log.warning("Dashboard is reachable from the network and has NO login: use only on a trusted Wi-Fi")

    # Detector time -> wall-clock time. Live sources are timestamped with time.monotonic(); files use the
    # video's own timeline, which we anchor to when processing started.
    if src.is_live:
        to_wall = lambda t: time.time() - (time.monotonic() - t)   # noqa: E731
    else:
        run_start = time.time()
        to_wall = lambda t: run_start + t                          # noqa: E731

    fps, last_wall = 0.0, None
    play_start = None   # (wall time, video time) for pacing file playback
    event_id = None     # database row of the event in progress
    last_out = None
    try:
        while True:
            item = src.read()
            if item is None:
                log.info("Source ended")
                break
            frame, t = item

            result = pose.process(frame, t)
            out = detector.update(t, result.keypoints)
            last_out = out

            if out.event_started:
                e = out.event_started
                log.warning("SEIZURE EVENT detected at t=%.1fs (rhythm since t=%.1fs)", e.detected_at, e.start)
                # Alarm first: it is local and instant. Everything network-related is queued in the background.
                alerts.on_event_start(to_wall(e.start), snapshot(frame, result, t, recorder, snapshot_privacy))
                clip = recorder.start_event(to_wall(e.start)) if recorder else None
                event_id = store.start_event(iso(to_wall(e.start)), e.confidence, clip, source_name)
            if out.event_ended:
                e = out.event_ended
                log.warning("Event ended: duration %.1fs, confidence %.2f", e.duration, e.confidence)
                alerts.on_event_end(to_wall(e.end))
                if recorder:
                    recorder.end_event(e.end)
                store.end_event(event_id, iso(to_wall(e.end)), e.duration, e.confidence)
                event_id = None
            if recorder:
                recorder.add_frame(frame, result, out, to_wall(t))
            if live:
                live.update(dashboard_frame(frame, result, out, to_wall(t), source_name, stream_privacy),
                            {"state": out.state.value, "source": source_name, "body_visible": out.body_visible,
                             "rhythm_s": round(out.rhythm_duration_s, 1), "onset_s": out.onset_s,
                             "confidence": round(out.current_event.confidence if out.current_event
                                                 else out.window_confidence, 2)})

            now = time.monotonic()
            if last_wall is not None:
                fps = 0.9 * fps + 0.1 / max(now - last_wall, 1e-6)   # smoothed for readability
            last_wall = now

            cv2.imshow("Seizure Monitor", render_live(frame, result, out, det_cfg, fps,
                                                      recording=bool(recorder and recorder.recording),
                                                      alert=alerts.status()))

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
            if key in (ord("s"), ord("S")):
                alerts.acknowledge_local()
    finally:
        # Stopped mid-event: close the record with what we know so it isn't left open forever.
        if event_id is not None and last_out is not None and last_out.current_event is not None:
            e = last_out.current_event
            log.warning("Monitor stopped during an event; recording its end as the stop time")
            store.end_event(event_id, iso(to_wall(last_out.t)), last_out.t - e.start, e.confidence)
        if live:
            live.running = False
        if server:
            server.stop()
        alerts.close()
        if recorder:
            recorder.close()
        src.release()
        pose.close()
        cv2.destroyAllWindows()


def dashboard_frame(frame, result, out, wall_time, source_name, privacy):
    """Frame for the web stream: skeleton + time/state banner (pixelated if dashboard.blur_stream)."""
    f = frame.copy()
    if privacy:
        f = privacy.apply(f, result.all_landmarks, out.t)
    draw_skeleton(f, result, out)
    draw_clip_banner(f, out, wall_time, source_name)
    return f


def snapshot(frame, result, t, recorder, privacy) -> bytes | None:
    """Still image for the alert, with the same privacy filtering as the clips."""
    if recorder and recorder.latest_jpeg:
        return recorder.latest_jpeg
    ok, jpeg = cv2.imencode(".jpg", privacy.apply(frame.copy(), result.all_landmarks, t))
    return jpeg.tobytes() if ok else None


if __name__ == "__main__":
    main()
