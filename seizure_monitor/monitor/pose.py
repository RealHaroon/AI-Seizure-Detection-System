"""MediaPipe pose estimation wrapper.

Uses the MediaPipe Tasks PoseLandmarker (mediapipe >= 0.10.30 removed the old
`mp.solutions.pose` API). It is a pretrained neural network that finds 33 body
landmarks per frame; we keep the ones the detector tracks.
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

log = logging.getLogger(__name__)

# MediaPipe landmark indices for the points the detector uses.
_MP_INDEX = {name: vision.PoseLandmark[name.upper()].value for name in TRACKED_POINTS}
SKELETON_CONNECTIONS = [(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS]


@dataclass
class PoseResult:
    keypoints: dict | None          # tracked point name -> (x_px, y_px, visibility); None if no person
    all_landmarks: list | None      # all 33 (x_px, y_px, visibility), for drawing the skeleton


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
    def __init__(self, cfg: dict, model_path: Path):
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            # VIDEO mode tracks the person between frames: faster and smoother than per-image detection.
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=cfg["min_detection_confidence"],
            min_tracking_confidence=cfg["min_tracking_confidence"],
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_ms = -1

    def process(self, frame_bgr, t: float) -> PoseResult:
        # MediaPipe requires strictly increasing integer millisecond timestamps.
        ms = max(int(t * 1000), self._last_ms + 1)
        self._last_ms = ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ms)
        if not result.pose_landmarks:
            return PoseResult(None, None)

        # Convert to pixels: normalised coords would stretch x vs y by the frame's aspect ratio.
        h, w = frame_bgr.shape[:2]
        lms = [(lm.x * w, lm.y * h, lm.visibility if lm.visibility is not None else 0.0)
               for lm in result.pose_landmarks[0]]
        return PoseResult({name: lms[i] for name, i in _MP_INDEX.items()}, lms)

    def close(self):
        self._landmarker.close()
