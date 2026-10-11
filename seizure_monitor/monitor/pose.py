"""MediaPipe pose estimation wrapper with patient selection.

Uses the MediaPipe Tasks PoseLandmarker (mediapipe >= 0.10.30 removed the old
`mp.solutions.pose` API). It is a pretrained neural network that finds 33 body
landmarks per person; we keep the ones the detector tracks.

Patient selection: on hospital videos a single-person tracker often locked onto
a nurse leaning over the bed. So we detect up to `num_poses` people and keep
the one who is mostly inside the bed area, preferring whoever was the patient
a moment ago (a nurse's head and arms may reach into the bed, but her torso
and legs don't).
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from .detector import TRACKED_POINTS
from .motion import bed_pixels

log = logging.getLogger(__name__)

# MediaPipe landmark indices for the points the detector uses.
_MP_INDEX = {name: vision.PoseLandmark[name.upper()].value for name in TRACKED_POINTS}
SKELETON_CONNECTIONS = [(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS]


@dataclass
class PoseResult:
    keypoints: dict | None          # tracked point name -> (x_px, y_px, visibility); None if no patient
    all_landmarks: list | None      # all 33 (x_px, y_px, visibility), for drawing the skeleton
    people: int = 0                 # people detected in the frame (patient + others)


def select_patient(poses: list, bed_px: tuple, prev_center, min_in_bed: float, max_jump_px: float,
                   min_visibility: float = 0.5):
    """Pick the patient among detected people. Pure function (testable without MediaPipe).

    poses:       list of 33-landmark lists [(x_px, y_px, visibility), ...]
    bed_px:      (x0, y0, x1, y1) bed area in pixels
    prev_center: (x, y) of the patient in the previous frame, or None
    Returns (index of the patient or None, its centre or None).
    """
    x0, y0, x1, y1 = bed_px
    best, best_key, best_center = None, None, None
    for i, lms in enumerate(poses):
        pts = [(x, y) for x, y, v in lms if v >= min_visibility]
        if len(pts) < 4:
            continue
        inside = sum(x0 <= x <= x1 and y0 <= y <= y1 for x, y in pts) / len(pts)
        if inside < min_in_bed:
            continue                       # mostly outside the bed: a visitor or caregiver
        center = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
        dist = None if prev_center is None else ((center[0] - prev_center[0]) ** 2 +
                                                 (center[1] - prev_center[1]) ** 2) ** 0.5
        # Prefer continuity with the previous patient; otherwise whoever is most inside the bed.
        continuous = dist is not None and dist <= max_jump_px
        key = (continuous, -dist if continuous else inside)
        if best_key is None or key > best_key:
            best, best_key, best_center = i, key, center
    return best, best_center


def ensure_model(cfg: dict, base_dir: Path) -> Path:
    """Download the pose model once (~9 MB) if it isn't there yet."""
    model_dir = base_dir / cfg["model_dir"]
    path = model_dir / f"pose_landmarker_{cfg['model']}.task"
    if not path.exists():
        url = cfg["model_url"].format(model=cfg["model"])
        log.info("Downloading pose model (one time) from %s", url)
        model_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    return path


class PoseEstimator:
    def __init__(self, cfg: dict, model_path: Path, bed_area=None):
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            # VIDEO mode tracks the person between frames: faster and smoother than per-image detection.
            running_mode=vision.RunningMode.VIDEO,
            num_poses=cfg["num_poses"],
            min_pose_detection_confidence=cfg["min_detection_confidence"],
            min_tracking_confidence=cfg["min_tracking_confidence"],
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_ms = -1
        self.bed_area = bed_area
        self.min_in_bed = cfg["min_in_bed_fraction"]
        self.max_jump = cfg["max_jump"]
        self._prev_center = None

    def process(self, frame_bgr, t: float) -> PoseResult:
        # MediaPipe requires strictly increasing integer millisecond timestamps.
        ms = max(int(t * 1000), self._last_ms + 1)
        self._last_ms = ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ms)
        if not result.pose_landmarks:
            return PoseResult(None, None, 0)       # keep _prev_center: the patient may reappear in place

        # Convert to pixels: normalised coords would stretch x vs y by the frame's aspect ratio.
        h, w = frame_bgr.shape[:2]
        poses = [[(lm.x * w, lm.y * h, lm.visibility if lm.visibility is not None else 0.0) for lm in person]
                 for person in result.pose_landmarks]
        bed = bed_pixels(self.bed_area, w, h)
        idx, center = select_patient(poses, bed, self._prev_center, self.min_in_bed,
                                     self.max_jump * (bed[2] - bed[0]))
        if center is not None:
            self._prev_center = center
        if idx is None:
            return PoseResult(None, None, len(poses))
        lms = poses[idx]
        return PoseResult({name: lms[i] for name, i in _MP_INDEX.items()}, lms, len(poses))

    def close(self):
        self._landmarker.close()
