"""Evaluate the detectors on labelled videos.

    python tools/evaluate.py VIDEO_FOLDER LABELS.csv [--detectors both|movement|rhythmic]
                             [--demo] [--tolerance 10] [--out results.csv]

labels CSV (one row per labelled seizure; normal videos get one row with empty times).
The optional bed_area column gives each video's bed region as "x0 y0 x1 y1" fractions of the
frame (cameras differ between videos); if empty, motion.bed_area from config.yaml is used.

    filename,seizure_start_s,seizure_end_s,bed_area
    S1.mp4,10,52,0.25 0.11 0.69 1.0
    pat05_Sz1PG.mp4,75,140,
    pat05_Sz1PG.mp4,600,655,     <- a second seizure in the same video
    N1.mp4,,,

Reports:
  - sensitivity: labelled seizures detected / labelled seizures (with a 95% confidence interval,
    because on a handful of videos the uncertainty is large and should be shown, not hidden)
  - detection latency: seconds from labelled seizure start to the moment an alert would fire
  - false alarms per hour of normal (seizure-free) footage
  - a per-video table, printed and written to CSV

Matching rule: a detection counts as catching a labelled seizure if the alert time falls within
[seizure_start - tolerance, seizure_end + tolerance] (default tolerance 10 s, because human
labels of onset/offset are imprecise). Any other detection is a false alarm.

Pose estimation and optical flow (the slow parts) are cached per video in data/eval_cache/, so
after the first run you can change detector thresholds in config.yaml and re-score in seconds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from monitor.detector import TRACKED_POINTS, Detector, DetectorConfig  # noqa: E402
from monitor.fusion import EventFuser  # noqa: E402
from monitor.motion import MotionConfig, MotionDetector  # noqa: E402

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


# ------------------------------------------------------------------ scoring (pure)

@dataclass
class Detection:
    detected_at: float     # when the alert would fire (seconds into the video)
    start: float           # when the movement began
    end: float | None
    trigger: str = ""      # which detector(s) fired: rhythmic / movement / rhythmic+movement


@dataclass
class VideoResult:
    filename: str
    duration_s: float
    seizures: list                          # [(start, end), ...]
    detections: list = field(default_factory=list)
    person_fraction: float = 0.0            # share of analysed frames where a person was found
    detected: int = 0                       # labelled seizures caught
    latencies: list = field(default_factory=list)
    false_alarms: int = 0


def load_labels(path: Path) -> dict[str, list]:
    """filename -> list of (start, end) seizures; empty list = normal video."""
    labels: dict[str, list] = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name = row["filename"].strip()
            if not name:
                continue
            labels.setdefault(name, [])
            s, e = row.get("seizure_start_s", "").strip(), row.get("seizure_end_s", "").strip()
            if s or e:
                if not (s and e):
                    raise ValueError(f"{name}: both seizure_start_s and seizure_end_s are needed")
                if float(e) <= float(s):
                    raise ValueError(f"{name}: seizure_end_s must be after seizure_start_s")
                labels[name].append((float(s), float(e)))
    return labels


def load_bed_areas(path: Path) -> dict[str, list | None]:
    """filename -> [x0, y0, x1, y1] from the optional bed_area column (None if not given)."""
    areas: dict[str, list | None] = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name, raw = row["filename"].strip(), (row.get("bed_area") or "").strip()
            if name and raw:
                vals = [float(x) for x in raw.replace(",", " ").split()]
                if len(vals) != 4:
                    raise ValueError(f"{name}: bed_area needs 4 numbers 'x0 y0 x1 y1'")
                areas[name] = vals
            elif name:
                areas.setdefault(name, None)
    return areas


def score_video(r: VideoResult, tolerance_s: float) -> VideoResult:
    """Match detections to labelled seizures (each detection can catch at most one seizure)."""
    used = set()
    r.detected, r.latencies = 0, []
    for s_start, s_end in sorted(r.seizures):
        for i, d in enumerate(r.detections):
            if i not in used and s_start - tolerance_s <= d.detected_at <= s_end + tolerance_s:
                used.add(i)
                r.detected += 1
                r.latencies.append(d.detected_at - s_start)
                break
    r.false_alarms = len(r.detections) - len(used)
    return r


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a proportion k/n (Wilson score; behaves well for small n)."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(results: list[VideoResult]) -> dict:
    n_seiz = sum(len(r.seizures) for r in results)
    caught = sum(r.detected for r in results)
    latencies = [x for r in results for x in r.latencies]
    normal = [r for r in results if not r.seizures]
    normal_h = sum(r.duration_s for r in normal) / 3600
    fa_normal = sum(r.false_alarms for r in normal)
    return {
        "videos": len(results),
        "seizures": n_seiz,
        "detected": caught,
        "sensitivity": caught / n_seiz if n_seiz else math.nan,
        "sensitivity_ci": wilson_interval(caught, n_seiz),
        "latency_median_s": float(np.median(latencies)) if latencies else math.nan,
        "latency_range_s": (min(latencies), max(latencies)) if latencies else (math.nan, math.nan),
        "normal_videos": len(normal),
        "normal_hours": normal_h,
        "false_alarms_normal": fa_normal,
        "false_alarms_per_hour": fa_normal / normal_h if normal_h > 0 else math.nan,
        "extra_detections_in_seizure_videos": sum(r.false_alarms for r in results if r.seizures),
    }


# --------------------------------------------------------------- video processing

def extract_track(video: Path, cfg: dict, bed_area, cache_dir: Path):
    """One decoding pass per video: pose keypoints AND bed-area optical flow, cached.

    Returns (times (n,), keypoints (n, 10, 3) NaN where no person,
             cells (n, grid*grid, 2) bed-area motion in bed widths/s (NaN on the first frame), duration_s).
    The cache key includes everything that changes these outputs, so stale caches are never reused.
    """
    mcfg = MotionConfig.from_dict({**cfg["motion"], "bed_area": bed_area})
    st = video.stat()
    key_src = repr((str(video.resolve()), st.st_size, int(st.st_mtime), cfg["pose"], cfg["capture"],
                    bed_area, mcfg.flow_width, mcfg.grid))
    cache = cache_dir / f"{video.stem}_{hashlib.sha1(key_src.encode()).hexdigest()[:12]}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["t"], d["kp"], d["cells"], float(d["duration"])

    from monitor.capture import VideoSource            # imported here: scoring works without OpenCV/MediaPipe
    from monitor.motion import BedFlow
    from monitor.pose import PoseEstimator, ensure_model
    pose = PoseEstimator(cfg["pose"], ensure_model(cfg["pose"], BASE_DIR))
    flow = BedFlow(mcfg)
    src = VideoSource(str(video), cfg["capture"]["width"], cfg["capture"]["process_fps"])
    import cv2
    total = src.cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(src.cap.get(cv2.CAP_PROP_FPS), 1e-6)
    n_cells = mcfg.grid * mcfg.grid
    times, kps, cells = [], [], []
    t0, next_report = time.time(), 0.1
    try:
        while (item := src.read()) is not None:
            frame, t = item
            res = pose.process(frame, t)
            row = np.full((len(TRACKED_POINTS), 3), np.nan)
            if res.keypoints:
                for i, name in enumerate(TRACKED_POINTS):
                    row[i] = res.keypoints[name]
            c = flow.process(frame, t)
            times.append(t)
            kps.append(row)
            cells.append(np.full((n_cells, 2), np.nan) if c is None else c)
            if total > 0 and t / total >= next_report:
                print(f"    {video.name}: {t / total:4.0%}  ({time.time() - t0:.0f} s elapsed)", flush=True)
                next_report += 0.1
    finally:
        src.release()
        pose.close()
    if not times:
        # Don't cache an empty result: it would hide the problem on every later run.
        raise RuntimeError("no frames could be decoded (unsupported codec such as AV1? "
                           "convert with: ffmpeg -i in.mp4 -c:v libx264 out.mp4)")
    t_arr = np.array(times)
    kp_arr = np.array(kps).reshape(-1, len(TRACKED_POINTS), 3)
    cell_arr = np.array(cells).reshape(-1, n_cells, 2)
    duration = float(t_arr[-1])
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, t=t_arr, kp=kp_arr, cells=cell_arr, duration=duration)
    return t_arr, kp_arr, cell_arr, duration


def run_detectors(times, kps, cells, det_cfg: DetectorConfig, mot_cfg: MotionConfig | None, demo: bool,
                  use: str = "both") -> list[Detection]:
    """Replay a cached track through the same detectors + fusion the live system uses.

    use: "both" (as deployed), "rhythmic" (pose detector only) or "movement" (bed-motion detector only).
    """
    det = Detector(det_cfg, demo=demo)
    mot = MotionDetector(mot_cfg, demo=demo) if (mot_cfg is not None and use != "rhythmic") else None
    fuser = EventFuser()
    found = []
    for i, t in enumerate(times):
        row = kps[i]
        kp = None
        if use != "movement" and not np.isnan(row[:, 0]).all():
            kp = {name: tuple(row[j]) for j, name in enumerate(TRACKED_POINTS)}
        pose_out = det.update(float(t), kp)
        mot_out = None
        if mot is not None:
            c = cells[i] if cells is not None else None
            mot_out = mot.update(float(t), None if c is None or np.isnan(c).any() else c)
        out = fuser.update(pose_out, mot_out)
        if out.event_started:
            e = out.event_started
            found.append(Detection(e.detected_at, e.start, None))
        if out.event_ended:
            found[-1].end = out.event_ended.end
        if found and out.current_event is not None:
            found[-1].trigger = out.current_event.trigger
        elif found and out.event_ended is not None:
            found[-1].trigger = out.event_ended.trigger
    return found


def run_detector(times, kps, det_cfg: DetectorConfig, demo: bool) -> list[Detection]:
    """Pose (rhythm) detector only."""
    return run_detectors(times, kps, None, det_cfg, None, demo, use="rhythmic")


# ------------------------------------------------------------------------ output

def _fmt(x, nd=1):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def print_report(results: list[VideoResult], s: dict, tolerance_s: float, settings: str):
    print("\nPer video")
    print(f"  {'file':<28} {'min':>6} {'person':>7} {'seizures':>9} {'caught':>7} {'latency_s':>10} "
          f"{'false_al':>9}  triggers")
    for r in results:
        lat = ", ".join(f"{x:.0f}" for x in r.latencies) or "–"
        trig = ", ".join(d.trigger for d in r.detections) or "–"
        print(f"  {r.filename[:28]:<28} {r.duration_s / 60:6.1f} {r.person_fraction:7.0%} "
              f"{len(r.seizures):9d} {r.detected:7d} {lat:>10} {r.false_alarms:9d}  {trig}")
    lo, hi = s["sensitivity_ci"]
    print("\nSummary")
    print(f"  Detectors              : {settings}")
    print(f"  Matching tolerance     : ±{tolerance_s:.0f} s around each labelled seizure")
    print(f"  Sensitivity            : {s['detected']}/{s['seizures']} seizures detected"
          + (f" = {s['sensitivity']:.0%} (95% CI {lo:.0%}–{hi:.0%})" if s["seizures"] else ""))
    print(f"  Detection latency      : median {_fmt(s['latency_median_s'])} s "
          f"(range {_fmt(s['latency_range_s'][0])} to {_fmt(s['latency_range_s'][1])} s)")
    print(f"  False alarms           : {s['false_alarms_normal']} in {s['normal_hours']:.2f} h of normal footage"
          + (f" = {s['false_alarms_per_hour']:.2f} per hour" if s["normal_hours"] > 0 else " (no normal videos)"))
    print(f"  Extra detections in seizure videos (outside labelled seizures): "
          f"{s['extra_detections_in_seizure_videos']}")
    if s["seizures"] < 20 or s["normal_hours"] < 8:
        print("  NOTE: small test set: treat these numbers as a first indication, not a validated result.")
    low_pose = [r.filename for r in results if r.person_fraction < 0.7]
    if low_pose:
        print(f"  NOTE: pose tracking found a person in <70% of frames for: {', '.join(low_pose)}"
              " (results there say more about tracking than about the detector)")


def write_csv(path: Path, results: list[VideoResult], s: dict):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "duration_min", "person_detected_pct", "labelled_seizures", "detected",
                    "latencies_s", "false_alarms", "detections_at_s"])
        for r in results:
            w.writerow([r.filename, f"{r.duration_s / 60:.2f}", f"{r.person_fraction * 100:.0f}", len(r.seizures),
                        r.detected, " ".join(f"{x:.1f}" for x in r.latencies), r.false_alarms,
                        " ".join(f"{d.detected_at:.1f}" for d in r.detections)])
        lo, hi = s["sensitivity_ci"]
        w.writerow([])
        w.writerow(["SUMMARY", "seizures", s["seizures"], "detected", s["detected"],
                    "sensitivity", _fmt(s["sensitivity"], 3), "ci95", f"{_fmt(lo, 3)}-{_fmt(hi, 3)}"])
        w.writerow(["", "median_latency_s", _fmt(s["latency_median_s"]), "normal_hours", f"{s['normal_hours']:.3f}",
                    "false_alarms_normal", s["false_alarms_normal"],
                    "false_alarms_per_hour", _fmt(s["false_alarms_per_hour"], 2)])


def main():
    ap = argparse.ArgumentParser(description="Evaluate the seizure detector on labelled videos")
    ap.add_argument("videos", type=Path, help="folder containing the videos")
    ap.add_argument("labels", type=Path, help="labels CSV: filename,seizure_start_s,seizure_end_s")
    ap.add_argument("--config", type=Path, default=BASE_DIR / "config.yaml")
    ap.add_argument("--demo", action="store_true", help="evaluate with the demo (shorter) onset settings")
    ap.add_argument("--detectors", choices=["both", "movement", "rhythmic"], default="both",
                    help="both = as deployed (default); or one detector alone, to compare them")
    ap.add_argument("--tolerance", type=float, default=10.0, help="matching tolerance in seconds (default 10)")
    ap.add_argument("--out", type=Path, help="results CSV (default: data/evaluation_results.csv)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    # Default output goes in the gitignored data folder: filenames of patient videos stay out of git.
    args.out = args.out or BASE_DIR / cfg["recording"]["data_dir"] / "evaluation_results.csv"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    det_cfg = DetectorConfig.from_dict(cfg["detector"])
    mot_base = MotionConfig.from_dict(cfg["motion"])
    labels = load_labels(args.labels)
    bed_areas = load_bed_areas(args.labels)
    cache_dir = BASE_DIR / cfg["recording"]["data_dir"] / "eval_cache"

    videos = {p.name: p for p in sorted(args.videos.iterdir()) if p.suffix.lower() in VIDEO_EXTS}
    for name in sorted(set(labels) - set(videos)):
        print(f"  WARNING: {name} is in the labels but not in {args.videos}; skipped")
    for name in sorted(set(videos) - set(labels)):
        print(f"  (not labelled, skipped: {name})")

    results = []
    for name in sorted(set(labels) & set(videos)):
        print(f"  processing {name} ...", flush=True)
        try:
            area = bed_areas.get(name) or cfg["motion"]["bed_area"]
            times, kps, cells, duration = extract_track(videos[name], cfg, area, cache_dir)
        except Exception as e:
            print(f"  ERROR reading {name}: {e}; skipped")
            continue
        r = VideoResult(name, duration, labels[name],
                        person_fraction=float(np.mean(~np.isnan(kps[:, 0, 0]))) if len(kps) else 0.0)
        mot_cfg = MotionConfig.from_dict({**cfg["motion"], "bed_area": area}) if mot_base.enabled else None
        r.detections = run_detectors(times, kps, cells, det_cfg, mot_cfg, args.demo, args.detectors)
        results.append(score_video(r, args.tolerance))

    if not results:
        print("No labelled videos found.")
        return
    s = summarize(results)
    onset = det_cfg.demo_onset_s if args.demo else det_cfg.onset_s
    sustain = mot_base.demo_sustain_s if args.demo else mot_base.sustain_s
    settings = {"both": f"movement (>= {mot_base.min_speed} bed widths/s for {sustain:.0f} s) OR rhythm ({onset:.0f} s)",
                "movement": f"movement only (>= {mot_base.min_speed} bed widths/s for {sustain:.0f} s)",
                "rhythmic": f"rhythm only ({onset:.0f} s of sustained rhythm)"}[args.detectors]
    print_report(results, s, args.tolerance, settings)
    write_csv(args.out, results, s)
    print(f"\n  Per-video results written to {args.out}")


if __name__ == "__main__":
    main()
