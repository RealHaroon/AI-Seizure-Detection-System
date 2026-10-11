"""Patient selection among several people, and the arm / pause switch."""

from types import SimpleNamespace

import numpy as np

from dashboard.app import LiveState, create_app
from monitor.control import ArmSwitch
from monitor.pose import select_patient
from monitor.recorder import EventRecorder
from monitor.storage import EventStore

BED = (200, 100, 400, 400)      # bed area in pixels: x0, y0, x1, y1


def person(cx, cy, w=60, h=200, vis=0.9):
    """33 fake landmarks spread over a w x h body centred at (cx, cy)."""
    rng = np.random.default_rng(int(cx * 7 + cy))
    return [(cx + rng.uniform(-w / 2, w / 2), cy + rng.uniform(-h / 2, h / 2), vis) for _ in range(33)]


def pick(poses, prev=None):
    return select_patient(poses, BED, prev, min_in_bed=0.6, max_jump_px=50)


def test_picks_person_in_bed_not_nurse_beside_it():
    nurse = person(470, 250)                      # standing right of the bed
    patient = person(300, 250)
    idx, _ = pick([nurse, patient])
    assert idx == 1


def test_nurse_leaning_in_is_not_the_patient():
    # Only her head/arms reach into the bed: most of her body is outside.
    nurse = [(x - 120 if i < 25 else x, y, v) for i, (x, y, v) in enumerate(person(520, 250))]
    patient = person(300, 250)
    assert pick([nurse, patient])[0] == 1
    assert pick([nurse])[0] is None               # nurse alone is never taken as the patient


def test_prefers_continuity_with_previous_patient():
    a, b = person(260, 250), person(350, 250)     # two people in the bed (e.g. parent lying beside child)
    _, centre_b = pick([b])
    idx, _ = pick([a, b], prev=centre_b)
    assert idx == 1                               # keeps following the same person


def test_nobody_detected():
    assert pick([]) == (None, None)


def test_arm_switch():
    s = ArmSwitch(True)
    assert s.armed and s.toggle() is False and not s.armed
    s.set(True)
    assert s.armed


def test_dashboard_pause_resume(tmp_path):
    switch = ArmSwitch(True)
    live = LiveState(70, alert_status=lambda: {"call": False}, switch=switch)
    app = create_app(EventStore(tmp_path / "e.db"), tmp_path, live)
    c = app.test_client()
    assert c.get("/api/status").get_json()["armed"] is True
    assert c.post("/api/arm", json={"armed": False}).get_json() == {"armed": False}
    assert not switch.armed and c.get("/api/status").get_json()["armed"] is False
    assert c.post("/api/arm", json={"armed": "yes"}).status_code == 400
    review_only = create_app(EventStore(tmp_path / "e.db"), tmp_path, None).test_client()
    assert review_only.post("/api/arm", json={"armed": True}).status_code == 404


def test_recorder_stop_now_finishes_clip(tmp_path):
    from monitor.detector import DetectorOutput, State
    rec = EventRecorder({"pre_event_s": 2, "post_event_s": 30, "fps": 15, "jpeg_quality": 80},
                        {"blur_person": False, "pixel_blocks": 24, "margin": 0.15, "hold_s": 2.0},
                        tmp_path / "clips", "cam")
    none = SimpleNamespace(all_landmarks=None, keypoints=None)
    frame = np.zeros((120, 160, 3), np.uint8)
    for i in range(45):
        t = i / 15
        if i == 30:
            rec.start_event(1e9 + t)
        rec.add_frame(frame, none, DetectorOutput(t=t, state=State.NORMAL, body_visible=False,
                                                  warming_up=False, window_rhythmic=False), 1e9 + t)
    assert rec.recording
    rec.stop_now()                                # pause: no 30 s post-event wait
    assert not rec.recording and len(rec._buffer) == 0
    rec.close()
    assert len(list((tmp_path / "clips").glob("*.mp4"))) == 1
