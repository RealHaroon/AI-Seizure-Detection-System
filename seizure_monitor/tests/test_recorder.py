"""Recorder tests with synthetic frames (short pre/post times to keep tests fast)."""

from types import SimpleNamespace

import cv2
import numpy as np

from monitor.detector import DetectorOutput, State
from monitor.recorder import EventRecorder, PrivacyFilter

FPS = 15
REC = {"pre_event_s": 3, "post_event_s": 2, "fps": FPS, "jpeg_quality": 80}
PRIV = {"blur_person": True, "pixel_blocks": 24, "margin": 0.15, "hold_s": 2.0}
NO_PERSON = SimpleNamespace(all_landmarks=None, keypoints=None)


def frame_at(i):
    f = np.zeros((480, 640, 3), np.uint8)
    cv2.putText(f, str(i), (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 8)
    return f


def output(t):
    return DetectorOutput(t=t, state=State.NORMAL, body_visible=False, warming_up=False, window_rhythmic=False)


def clip_seconds(path):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return n / fps


def test_clip_contains_pre_and_post_event_footage(tmp_path):
    rec = EventRecorder(REC, {**PRIV, "blur_person": False}, tmp_path / "clips", "test cam")
    start_t, end_t = 10.0, 14.0
    clip = None
    for i in range(int(25 * FPS)):
        t = i / FPS
        if clip is None and t >= start_t:
            clip = rec.start_event(wall_time=1e9 + t)
        if abs(t - end_t) < 1e-9:
            rec.end_event(end_t)
        rec.add_frame(frame_at(i), NO_PERSON, output(t), wall_time=1e9 + t)
    rec.close()

    path = tmp_path / clip
    assert path.exists()
    # pre (3 s) + event (10 -> 14 s) + post (2 s) = 9 s of footage
    assert abs(clip_seconds(path) - 9.0) < 0.3


def test_buffer_is_bounded(tmp_path):
    rec = EventRecorder(REC, PRIV, tmp_path / "clips", "test cam")
    for i in range(int(60 * FPS)):
        rec.add_frame(frame_at(i), NO_PERSON, output(i / FPS), wall_time=1e9 + i / FPS)
    assert len(rec._buffer) <= REC["pre_event_s"] * FPS + 1
    assert not rec.recording


def test_privacy_pixelates_person_box_only():
    pf = PrivacyFilter(PRIV)
    frame = (np.random.default_rng(0).random((480, 640, 3)) * 255).astype(np.uint8)
    original = frame.copy()
    # A "person" occupying x 200-300, y 100-400.
    lms = [(200, 100, 1.0), (300, 400, 1.0)]
    out = pf.apply(frame, lms, t=0.0)
    assert not np.array_equal(out[150:350, 220:280], original[150:350, 220:280])   # person changed
    assert np.array_equal(out[:, :150], original[:, :150])                           # background untouched


def test_privacy_fail_safe_blurs_whole_frame_when_person_lost():
    pf = PrivacyFilter(PRIV)
    frame = (np.random.default_rng(1).random((480, 640, 3)) * 255).astype(np.uint8)
    original = frame.copy()
    out = pf.apply(frame, None, t=0.0)   # never seen a person
    assert not np.array_equal(out[:, :150], original[:, :150])
