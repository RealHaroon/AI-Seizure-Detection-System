"""Detector A (bed-area movement), the fusion of both detectors, and the bed-area tool."""

import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from monitor.detector import DetectorOutput, Event, State
from monitor.fusion import EventFuser
from monitor.motion import BedFlow, MotionConfig, MotionDetector, MotionOutput

CFG_PATH = Path(__file__).parent.parent / "config.yaml"
MCFG = yaml.safe_load(CFG_PATH.read_text())["motion"]
FPS = 15


def mcfg(**over):
    return MotionConfig.from_dict({**MCFG, **over})


def run_motion(speed_fn, duration, seed=0, **over):
    """speed_fn(t) -> speed of the moving cells (bed widths/s); other cells get small noise."""
    rng = np.random.default_rng(seed)
    cfg = mcfg(**over)
    det = MotionDetector(cfg)
    started, ended = [], []
    for t in np.arange(0, duration, 1 / FPS):
        cells = rng.normal(0, 0.003, (cfg.grid * cfg.grid, 2))          # camera noise / breathing
        cells[:6] += [speed_fn(t), 0]                                    # the moving part of the bed
        out = det.update(float(t), cells)
        if out.event_started:
            started.append(out.event_started)
        if out.event_ended:
            ended.append(out.event_ended)
    return started, ended


def test_calm_bed_no_event():
    started, _ = run_motion(lambda t: 0.0, 120)
    assert started == []


def test_sustained_vigorous_movement_detected():
    speed = lambda t: 0.12 if 10 <= t < 40 else 0.0
    started, ended = run_motion(speed, 60)
    assert len(started) == 1 and len(ended) == 1
    ev = ended[0]
    assert 9.5 <= ev.start <= 11                                          # began when movement began
    # earliest alert: the window is sustain_fraction full, i.e. sustain_s x sustain_fraction after onset
    assert ev.detected_at - 10 == pytest.approx(MCFG["sustain_s"] * MCFG["sustain_fraction"], abs=0.5)
    assert 39 <= ev.end <= 40.5
    assert 0 < ev.confidence <= 1


def test_brief_burst_not_detected():
    started, _ = run_motion(lambda t: 0.12 if 10 <= t < 12.5 else 0.0, 40)
    assert started == []


def test_movement_with_short_pauses_still_detected():
    # Real convulsions pause briefly: 0.8 s moving, 0.2 s still, repeatedly.
    started, _ = run_motion(lambda t: 0.12 if (10 <= t < 40 and (t % 1.0) < 0.8) else 0.0, 60)
    assert len(started) == 1


def test_slow_movement_below_threshold_not_detected():
    started, _ = run_motion(lambda t: 0.6 * MCFG["min_speed"], 60)
    assert started == []


def test_config_validation():
    with pytest.raises(ValueError):
        mcfg(bed_area=[0.5, 0.1, 0.4, 0.9])          # x1 < x0
    with pytest.raises(ValueError):
        mcfg(top_cells=99)
    with pytest.raises(ValueError):
        MotionConfig.from_dict({**MCFG, "min_sped": 0.1})


# ---------------------------------------------------------------- optical flow

def _textured(h, w, seed):
    return (np.random.default_rng(seed).random((h, w, 3)) * 255).astype(np.uint8)


def test_bedflow_sees_movement_inside_bed_only():
    cfg = mcfg(bed_area=[0.5, 0.0, 1.0, 1.0])        # right half of the frame is the bed
    flow = BedFlow(cfg)
    bg = _textured(240, 320, 0)
    patch = _textured(60, 60, 1)
    inside, outside = [], []
    for case, x_of in (("inside", lambda i: 200 + 3 * (i % 2)), ("outside", lambda i: 40 + 3 * (i % 2))):
        flow = BedFlow(cfg)
        for i in range(10):
            f = bg.copy()
            x = x_of(i)
            f[90:150, x:x + 60] = patch                  # a textured "limb" jerking 3 px back and forth
            c = flow.process(f, i / FPS)
            if c is not None:
                (inside if case == "inside" else outside).append(np.linalg.norm(c, axis=1).max())
    assert np.median(inside) > MCFG["min_speed"]       # bed widths per second, comfortably above threshold
    assert np.median(outside) < 0.2 * MCFG["min_speed"]


# --------------------------------------------------------------------- fusion

def pose_out(t, event=None, ended=None):
    return DetectorOutput(t=t, state=State.SEIZURE_EVENT if event else State.NORMAL, body_visible=True,
                          warming_up=False, window_rhythmic=False, current_event=event, event_ended=ended)


def mot_out(t, event=None, ended=None, building=False):
    from monitor.motion import MotionEvent  # noqa: F401
    return MotionOutput(t=t, current_event=event, event_ended=ended, building=building)


def test_fusion_movement_then_rhythm_is_one_event():
    from monitor.motion import MotionEvent
    fz = EventFuser()
    m_ev = MotionEvent(start=10, detected_at=15, confidence=0.4)
    p_ev = Event(start=12, detected_at=20, confidence=0.7)
    o = fz.update(pose_out(15), mot_out(15, m_ev))
    assert o.event_started and o.event_started.start == 10 and o.event_started.trigger == "movement"
    o = fz.update(pose_out(20, p_ev), mot_out(20, m_ev))
    assert o.event_started is None and o.current_event.trigger == "rhythmic+movement"
    assert o.current_event.confidence == 0.7
    m_ev.end = 30
    o = fz.update(pose_out(35, p_ev), mot_out(35, None, ended=m_ev))   # movement over, rhythm continues
    assert o.event_ended is None and o.state == State.SEIZURE_EVENT
    p_ev.end = 38
    o = fz.update(pose_out(43, None, ended=p_ev), mot_out(43))          # both quiet now
    assert o.event_ended is not None and o.event_ended.end == 38 and o.state == State.NORMAL


def test_fusion_states():
    fz = EventFuser()
    assert fz.update(pose_out(1), mot_out(1, building=True)).state == State.MOVEMENT
    assert fz.update(pose_out(2), None).state == State.NORMAL            # motion detector disabled


# ------------------------------------------------------------- bed-area tool

def test_set_bed_area_keeps_config_comments(tmp_path):
    from tools.set_bed_area import write_bed_area
    p = tmp_path / "config.yaml"
    shutil.copy(CFG_PATH, p)
    before = p.read_text()
    write_bed_area(p, [0.25, 0.1, 0.7, 1.0])
    after = p.read_text()
    assert yaml.safe_load(after)["motion"]["bed_area"] == [0.25, 0.1, 0.7, 1.0]
    assert after.count("#") == before.count("#")                         # no comments lost
    assert len(after.splitlines()) == len(before.splitlines())
