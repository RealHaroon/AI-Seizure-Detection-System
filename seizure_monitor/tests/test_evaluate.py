import math

import numpy as np
import pytest

from tests.test_detector import BASE, CONFIG, LIMB_POINTS, oscillation, still
from monitor.detector import TRACKED_POINTS, DetectorConfig
from tools.evaluate import (Detection, VideoResult, load_labels, run_detector, score_video, summarize,
                            wilson_interval)


def det(at):
    return Detection(detected_at=at, start=at - 8, end=at + 20)


def test_load_labels(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("filename,seizure_start_s,seizure_end_s\nS1.mp4,10,52\nS1.mp4,300,340\nN1.mp4,,\n")
    assert load_labels(p) == {"S1.mp4": [(10, 52), (300, 340)], "N1.mp4": []}
    p.write_text("filename,seizure_start_s,seizure_end_s\nS1.mp4,10,\n")
    with pytest.raises(ValueError):
        load_labels(p)


def test_match_within_tolerance_and_latency():
    r = score_video(VideoResult("S1.mp4", 120, [(10, 52)], detections=[det(19)]), tolerance_s=10)
    assert (r.detected, r.false_alarms, r.latencies) == (1, 0, [9])


def test_detection_outside_seizure_is_false_alarm():
    r = score_video(VideoResult("S1.mp4", 600, [(10, 52)], detections=[det(19), det(400)]), tolerance_s=10)
    assert (r.detected, r.false_alarms) == (1, 1)


def test_missed_seizure():
    r = score_video(VideoResult("S1.mp4", 120, [(10, 52)], detections=[det(70)]), tolerance_s=10)
    assert (r.detected, r.false_alarms) == (0, 1)


def test_one_detection_cannot_catch_two_seizures():
    r = score_video(VideoResult("S.mp4", 300, [(10, 40), (45, 80)], detections=[det(42)]), tolerance_s=10)
    assert (r.detected, r.false_alarms) == (1, 0)


def test_summary_false_alarms_counted_over_all_seizure_free_time():
    seiz = score_video(VideoResult("S1.mp4", 600, [(10, 52)], detections=[det(19), det(400)]), 10)
    normal = score_video(VideoResult("N1.mp4", 1800, [], detections=[det(100)]), 10)
    s = summarize([seiz, normal])
    assert s["sensitivity"] == 1.0
    # Seizure-free time = 1800 s normal video + (600 - (0..62)) s of the seizure video = 2338 s; 2 false alarms
    assert s["false_alarms_normal"] == 2
    assert math.isclose(s["normal_hours"], 2338 / 3600)
    assert math.isclose(s["false_alarms_per_hour"], 2 / (2338 / 3600))
    assert s["extra_detections_in_seizure_videos"] == 1


def test_wilson_interval_small_sample_is_wide():
    lo, hi = wilson_interval(3, 5)
    assert 0.22 < lo < 0.24 and 0.87 < hi < 0.89      # 3/5 = 60%, but anywhere from ~23% to ~88%


def _track(motion, duration, fps=15, seed=0):
    """Synthetic cached-keypoint arrays, the same format keypoint_track() produces from a video."""
    rng = np.random.default_rng(seed)
    times = np.arange(0, duration, 1 / fps)
    kps = np.zeros((len(times), len(TRACKED_POINTS), 3))
    for j, t in enumerate(times):
        for i, name in enumerate(TRACKED_POINTS):
            dx, dy = motion(t, name, rng)
            kps[j, i] = (BASE[name][0] + dx + rng.normal(0, 1), BASE[name][1] + dy + rng.normal(0, 1), 0.9)
    return times, kps


def test_end_to_end_on_synthetic_keypoints():
    cfg = DetectorConfig.from_dict(CONFIG)
    times, kps = _track(oscillation(3.0, 15, LIMB_POINTS, t_on=10, t_off=40), duration=70)
    r = score_video(VideoResult("synthetic.mp4", 70, [(10, 40)], detections=run_detector(times, kps, cfg, False)), 10)
    assert (r.detected, r.false_alarms) == (1, 0)
    assert CONFIG["onset_s"] <= r.latencies[0] <= CONFIG["onset_s"] + 3

    times, kps = _track(still, duration=120)
    kps[200:260] = np.nan                                 # person briefly out of view
    assert run_detector(times, kps, cfg, False) == []


def test_repeat_detection_inside_caught_seizure_is_not_a_false_alarm():
    r = score_video(VideoResult("S.mp4", 300, [(50, 150)], detections=[det(60), det(120), det(250)]), tolerance_s=10)
    assert (r.detected, r.repeats, r.false_alarms) == (1, 1, 1)     # 120 s repeats; 250 s is a real false alarm
