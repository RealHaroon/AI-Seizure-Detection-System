import csv
import io

import numpy as np
import pytest

from dashboard.app import LiveState, create_app
from monitor.storage import EventStore


@pytest.fixture
def setup(tmp_path):
    store = EventStore(tmp_path / "events.db")
    (tmp_path / "clips").mkdir()
    (tmp_path / "clips" / "event_1.mp4").write_bytes(b"\x00" * 5000)
    e1 = store.start_event("2026-10-03T02:14:05", 0.61, "clips/event_1.mp4", "Bedroom camera")
    store.end_event(e1, "2026-10-03T02:15:10", 65.0, 0.72)
    e2 = store.start_event("2026-10-04T23:01:00", 0.40, None, "Bedroom camera")   # still in progress
    return store, tmp_path, e1, e2


def client(setup, live=None):
    store, data_dir, *_ = setup
    app = create_app(store, data_dir, live)
    app.testing = True
    return app.test_client()


def test_pages_render(setup):
    c = client(setup)
    _, _, e1, e2 = setup
    assert c.get("/").status_code == 200
    page = c.get("/events").get_data(as_text=True)
    assert "1 min 5 sec" in page and "In progress" in page and "Needs review" in page
    detail = c.get(f"/events/{e1}").get_data(as_text=True)
    assert "<video" in detail and "Real seizure" in detail
    assert c.get("/events/999").status_code == 404


def test_review_saves_label_and_note(setup):
    store, _, e1, _ = setup
    c = client(setup)
    r = c.post(f"/events/{e1}/review", data={"label": "real_seizure", "note": "GTC, matches history"})
    assert r.status_code == 302
    ev = store.get_event(e1)
    assert (ev["review_label"], ev["reviewer_note"]) == ("real_seizure", "GTC, matches history")
    assert c.post(f"/events/{e1}/review", data={"label": "maybe"}).status_code == 400


def test_csv_export(setup):
    store, _, e1, _ = setup
    store.set_review(e1, "false_alarm", "patient was brushing teeth, سلام")
    c = client(setup)
    r = c.get("/export.csv")
    assert r.status_code == 200 and "attachment" in r.headers["Content-Disposition"]
    rows = list(csv.DictReader(io.StringIO(r.data.decode("utf-8-sig"))))
    assert [row["event_id"] for row in rows] == [str(e1), str(setup[3])]       # oldest first
    assert rows[0]["duration_s"] == "65" and rows[0]["review_label"] == "false_alarm"
    assert "سلام" in rows[0]["reviewer_note"]                                  # Urdu/Arabic text survives


def test_clip_serving_and_traversal_blocked(setup):
    c = client(setup)
    r = c.get("/clips/event_1.mp4", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.data) == 100                         # seeking works
    assert c.get("/clips/../events.db").status_code == 404


def test_api_status(setup):
    assert client(setup).get("/api/status").get_json() == {"monitor_running": False, "call": False}
    live = LiveState(70, alert_status=lambda: {"call": True, "acknowledged_by": None})
    live.update(np.zeros((48, 64, 3), np.uint8), {"state": "SEIZURE EVENT"})
    s = client(setup, live).get("/api/status").get_json()
    assert s["monitor_running"] and s["state"] == "SEIZURE EVENT" and s["call"] is True
