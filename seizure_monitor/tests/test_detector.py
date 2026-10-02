"""Detector tests on synthetic keypoint trajectories (no camera needed).

A fake person stands still with realistic pose-estimation jitter; we then add
different kinds of movement to selected points and check what the detector says.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from monitor.detector import TRACKED_POINTS, Detector, DetectorConfig, State

CONFIG = yaml.safe_load((Path(__file__).parent.parent / "config.yaml").read_text())["detector"]

# Resting pose in pixels; shoulders 100 px apart, so 10 px of movement = 0.1 shoulder widths.
BASE = {
    "left_shoulder": (270, 150), "right_shoulder": (370, 150),
    "left_elbow": (250, 230), "right_elbow": (390, 230),
    "left_wrist": (245, 300), "right_wrist": (395, 300),
    "left_knee": (290, 400), "right_knee": (350, 400),
    "left_ankle": (290, 470), "right_ankle": (350, 470),
}
LIMB_POINTS = ["left_wrist", "left_elbow", "right_wrist", "right_elbow",
               "left_knee", "left_ankle", "right_knee", "right_ankle"]
JITTER_PX = 1.0   # typical frame-to-frame MediaPipe jitter for a still person


def make_detector(**overrides):
    return Detector(DetectorConfig.from_dict({**CONFIG, **overrides}))


def frame_times(duration, fps, rng, jitter=0.2):
    """Timestamps with realistic frame-interval jitter (cameras are never perfectly regular)."""
    dt = (1.0 / fps) * (1 + rng.uniform(-jitter, jitter, int(duration * fps)))
    return np.cumsum(dt)


def run(motion, duration=40.0, fps=15, seed=0, hidden=None, onset_s=None):
    """Feed the detector a synthetic recording.

    motion(t, name, rng) -> (dx, dy) pixel offset added to the resting pose.
    hidden(t) -> True if the whole body is out of view at time t.
    Returns (list of started events, list of ended events, list of states).
    """
    rng = np.random.default_rng(seed)
    det = make_detector(**({"onset_s": onset_s} if onset_s else {}))
    started, ended, states = [], [], []
    for t in frame_times(duration, fps, rng):
        if hidden and hidden(t):
            kp = None
        else:
            kp = {}
            for name in TRACKED_POINTS:
                dx, dy = motion(t, name, rng)
                x0, y0 = BASE[name]
                kp[name] = (x0 + dx + rng.normal(0, JITTER_PX), y0 + dy + rng.normal(0, JITTER_PX), 0.9)
        out = det.update(float(t), kp)
        states.append(out.state)
        if out.event_started:
            started.append(out.event_started)
        if out.event_ended:
            ended.append(out.event_ended)
    return started, ended, states


def oscillation(freq, amp_px, points, t_on=0.0, t_off=1e9):
    """Rhythmic jerking on the given points between t_on and t_off."""
    phases = {p: i * 0.7 for i, p in enumerate(points)}   # limbs don't jerk perfectly in sync

    def motion(t, name, rng):
        if name in points and t_on <= t < t_off:
            s = np.sin(2 * np.pi * freq * t + phases[name])
            return amp_px * s, 0.6 * amp_px * s
        return 0.0, 0.0
    return motion


def still(t, name, rng):
    return 0.0, 0.0


# ----------------------------------------------------------------- tests

def test_still_person_not_detected():
    started, _, states = run(still, duration=60)
    assert started == []
    assert State.RHYTHMIC not in states


def test_3hz_on_several_limbs_detected():
    # 5 s still, 25 s of 3 Hz jerking on all four limbs, then 20 s still.
    started, ended, _ = run(oscillation(3.0, 15, LIMB_POINTS, t_on=5, t_off=30), duration=50)
    assert len(started) == 1 and len(ended) == 1
    ev = ended[0]
    assert 5 <= ev.start <= 8                       # rhythm shows up within ~half a window of real onset
    assert ev.detected_at - ev.start == pytest.approx(CONFIG["onset_s"], abs=0.5)
    assert 28 <= ev.end <= 33                       # last rhythmic window is shortly after real offset
    assert 0.3 < ev.confidence <= 1.0


def test_random_noise_not_detected():
    # Large random movement on every point: plenty of amplitude, but no rhythm.
    def noise(t, name, rng):
        return rng.normal(0, 10), rng.normal(0, 10)
    started, _, _ = run(noise, duration=120)
    assert started == []


def test_slow_movement_not_detected():
    # 0.3 Hz large movement on all limbs, like turning over or stretching.
    started, _, _ = run(oscillation(0.3, 60, LIMB_POINTS + ["left_shoulder", "right_shoulder"]), duration=60)
    assert started == []


def test_brief_burst_not_detected():
    # 2 s of strong 3 Hz shaking (e.g. a shiver or shaking out a hand), well under onset_s.
    started, _, _ = run(oscillation(3.0, 15, LIMB_POINTS, t_on=10, t_off=12), duration=30)
    assert started == []


def test_single_limb_not_detected():
    # One rhythmic arm (e.g. brushing teeth, waving) must not trigger: we require several limbs.
    started, _, _ = run(oscillation(3.0, 15, ["right_wrist", "right_elbow"]), duration=40)
    assert started == []


@pytest.mark.parametrize("fps", [10, 30])
def test_independent_of_camera_fps(fps):
    started, ended, _ = run(oscillation(3.0, 15, LIMB_POINTS, t_on=5, t_off=30), duration=50, fps=fps)
    assert len(started) == 1 and len(ended) == 1
    assert 5 <= started[0].start <= 8


def test_body_lost_mid_event_ends_cleanly():
    # Seizure, then person leaves the frame for 15 s (e.g. falls out of view), then is back and still.
    motion = oscillation(3.0, 15, LIMB_POINTS, t_on=2, t_off=20)
    started, ended, states = run(motion, duration=50, hidden=lambda t: 20 <= t < 35)
    assert len(started) == 1 and len(ended) == 1   # one event, no spurious second event on return
    assert states[-1] == State.NORMAL


def test_demo_onset_is_faster():
    started, _, _ = run(oscillation(3.0, 15, LIMB_POINTS, t_on=5, t_off=30), duration=40, onset_s=3.0)
    assert started[0].detected_at - started[0].start == pytest.approx(3.0, abs=0.5)


def test_config_typo_rejected():
    with pytest.raises(ValueError):
        DetectorConfig.from_dict({**CONFIG, "onset_sec": 8})
