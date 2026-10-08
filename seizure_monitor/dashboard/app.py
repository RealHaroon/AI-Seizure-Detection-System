"""Web dashboard: live view, event list, clip review/labelling, CSV seizure diary.

Normally started by main.py in a background thread, so it can show the live
camera and current detector state. It can also run on its own to review
recorded events without a camera:

    python dashboard/app.py        # review mode, http://127.0.0.1:8000

Security: binds to 127.0.0.1 (this computer only) by default and never runs
in debug mode. There is no login, so only expose it on a trusted network.
"""

from __future__ import annotations

import csv
import io
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    # Run directly as a script: make the project root importable.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from flask import Flask, Response, abort, jsonify, redirect, render_template, request, send_from_directory, url_for

from monitor.fusion import TRIGGER_TEXT
from monitor.storage import REVIEW_LABELS, EventStore

LABEL_TEXT = {"unreviewed": "Needs review", "real_seizure": "Real seizure", "false_alarm": "False alarm"}


class LiveState:
    """Latest annotated frame + detector state, shared between the monitor loop and web requests."""

    def __init__(self, jpeg_quality: int, alert_status=None):
        self._lock = threading.Lock()
        self._frame = None
        self._frame_id = 0
        self._info: dict = {}
        self.jpeg_quality = jpeg_quality
        self.alert_status = alert_status     # fn() -> dict from AlertManager
        self.running = True

    def update(self, frame_bgr, info: dict):
        # Only stores a reference; JPEG encoding happens in the web thread, never in the detection loop.
        with self._lock:
            self._frame = frame_bgr
            self._frame_id += 1
            self._info = info

    def jpeg(self):
        with self._lock:
            frame, fid = self._frame, self._frame_id
        if frame is None:
            return None, fid
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        return (buf.tobytes() if ok else None), fid

    def status(self) -> dict:
        with self._lock:
            info = dict(self._info)
        alerts = self.alert_status() if self.alert_status else {}
        return {"monitor_running": self.running, **info, **alerts}


def create_app(store: EventStore, data_dir: Path, live: LiveState | None = None, stream_fps: float = 10) -> Flask:
    app = Flask(__name__)
    data_dir = Path(data_dir).resolve()

    # ------------------------------------------------------------ helpers

    def view(ev: dict) -> dict:
        """Add display-friendly fields to a database row."""
        start = datetime.fromisoformat(ev["start_time"])
        ev = dict(ev)
        ev["date"] = start.strftime("%a %d %b %Y")
        ev["time"] = start.strftime("%H:%M:%S")
        ev["duration"] = _fmt_duration(ev["duration_s"]) if ev["end_time"] else "In progress"
        ev["confidence_text"] = f"{ev['confidence']:.2f}" if ev["confidence"] is not None else "–"
        ev["label_text"] = LABEL_TEXT[ev["review_label"]]
        ev["trigger_text"] = TRIGGER_TEXT.get(ev.get("trigger") or "", "–")
        clip = data_dir / ev["clip_path"] if ev["clip_path"] else None
        ev["clip_available"] = bool(clip and clip.exists())
        return ev

    # -------------------------------------------------------------- pages

    @app.get("/")
    def live_page():
        recent = [view(e) for e in store.list_events()[:5]]
        return render_template("live.html", live=live is not None, recent=recent, page="live")

    @app.get("/events")
    def events_page():
        events = [view(e) for e in store.list_events()]
        counts = {k: sum(e["review_label"] == k for e in events) for k in REVIEW_LABELS}
        return render_template("events.html", events=events, counts=counts, page="events")

    @app.get("/events/<int:event_id>")
    def event_page(event_id):
        ev = store.get_event(event_id)
        if ev is None:
            abort(404)
        return render_template("event.html", ev=view(ev), labels=LABEL_TEXT, saved=request.args.get("saved"),
                               page="events")

    @app.post("/events/<int:event_id>/review")
    def review(event_id):
        if store.get_event(event_id) is None:
            abort(404)
        label = request.form.get("label", "")
        if label not in REVIEW_LABELS:
            abort(400, "Unknown label")
        note = request.form.get("note", "").strip()[:2000]
        store.set_review(event_id, label, note)
        return redirect(url_for("event_page", event_id=event_id, saved=1))

    @app.get("/export.csv")
    def export_csv():
        out = io.StringIO()
        w = csv.writer(out)
        w.writerow(["event_id", "date", "start_time", "end_time", "duration_s", "confidence_heuristic",
                    "detected_by", "review_label", "reviewer_note", "camera"])
        for e in reversed(store.list_events()):     # oldest first, like a diary
            start = datetime.fromisoformat(e["start_time"])
            end = datetime.fromisoformat(e["end_time"]).strftime("%H:%M:%S") if e["end_time"] else ""
            w.writerow([e["id"], start.strftime("%Y-%m-%d"), start.strftime("%H:%M:%S"), end,
                        f"{e['duration_s']:.0f}" if e["duration_s"] is not None else "",
                        f"{e['confidence']:.2f}" if e["confidence"] is not None else "",
                        TRIGGER_TEXT.get(e.get("trigger") or "", ""),
                        e["review_label"], e["reviewer_note"], e["source"]])
        # utf-8-sig adds a BOM so Excel shows non-English notes (e.g. Urdu) correctly.
        filename = f"seizure_diary_{datetime.now():%Y%m%d}.csv"
        return Response(out.getvalue().encode("utf-8-sig"), mimetype="text/csv",
                        headers={"Content-Disposition": f"attachment; filename={filename}"})

    @app.get("/clips/<path:name>")
    def clip(name):
        # send_from_directory refuses paths outside the folder and supports Range requests (video seeking).
        return send_from_directory(data_dir / "clips", name, mimetype="video/mp4")

    # ---------------------------------------------------------------- API

    @app.get("/api/status")
    def api_status():
        """Polled by the live page and by the Android automation app.

        The app should place a phone call when "call" is true. Use
        "incident_started" to avoid calling twice for the same incident.
        """
        if live is None:
            return jsonify({"monitor_running": False, "call": False})
        return jsonify(live.status())

    @app.get("/stream.mjpg")
    def stream():
        if live is None:
            abort(404)

        def frames():
            # MJPEG: an endless multipart response; the browser shows each part as the next frame.
            last_id = -1
            while live.running:
                jpg, fid = live.jpeg()
                if jpg is not None and fid != last_id:
                    last_id = fid
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                time.sleep(1.0 / stream_fps)

        return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.errorhandler(404)
    def not_found(e):
        return render_template("message.html", title="Not found", text="That page or event doesn't exist."), 404

    return app


class DashboardServer:
    """Runs the Flask app in a background thread (never debug mode)."""

    def __init__(self, app: Flask, host: str, port: int):
        from werkzeug.serving import make_server
        self._server = make_server(host, port, app, threaded=True)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True, name="dashboard")

    def start(self):
        self._thread.start()

    def stop(self):
        self._server.shutdown()


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "–"
    m, s = divmod(int(round(seconds)), 60)
    return f"{m} min {s} sec" if m else f"{s} sec"


if __name__ == "__main__":
    # Review-only mode: browse and label recorded events without running the camera.
    import yaml
    base = Path(__file__).resolve().parent.parent
    cfg = yaml.safe_load((base / "config.yaml").read_text())
    data = base / cfg["recording"]["data_dir"]
    dcfg = cfg["dashboard"]
    print(f"Review mode (no camera): http://{dcfg['host']}:{dcfg['port']}")
    create_app(EventStore(data / "events.db"), data).run(host=dcfg["host"], port=dcfg["port"], debug=False)
