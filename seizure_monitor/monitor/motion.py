"""Detector A: sustained intense movement inside the bed area (no pose needed).

Why this exists: on real hospital seizure videos the pose detector failed.
Blankets hid the legs, and pose tracking often locked onto a nurse instead of
the patient. Most of the convulsive movement was also large and irregular
rather than a clean 1.5-6 Hz rhythm. What did separate seizures from normal
footage very clearly was the *strength* of movement inside the bed: 10-35x
higher during seizures than in normal footage.

How it works:
  1. BedFlow: in each frame, crop the bed area, compute dense optical flow (how
     every pixel moved since the previous frame), split the bed into a grid and
     take the average motion of each cell. Speeds are in "bed widths per
     second", so they don't depend on resolution, distance or frame rate.
  2. MotionDetector (pure logic): intensity = mean speed of the few most-moving
     cells (convulsions may involve only part of the bed). An event starts when
     intensity stays above a threshold for most of `sustain_s`, and ends after
     `end_hold_s` of calm.

It sees movement through blankets and ignores people standing beside the bed.
Its weakness: any vigorous movement in bed (e.g. getting up, someone making
the bed) can also trigger it, which is why false alarms must be measured on
normal footage.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, fields

import numpy as np


@dataclass
class MotionConfig:
    enabled: bool
    bed_area: list | None      # [x0, y0, x1, y1] as fractions of the frame; None = whole frame
    flow_width: int
    grid: int
    top_cells: int
    min_speed: float
    sustain_s: float
    demo_sustain_s: float
    sustain_fraction: float
    end_hold_s: float
    confidence_full_speed: float

    @classmethod
    def from_dict(cls, d: dict) -> "MotionConfig":
        names = {f.name for f in fields(cls)}
        unknown, missing = set(d) - names, names - set(d)
        if unknown or missing:
            raise ValueError(f"motion config: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
        cfg = cls(**d)
        if cfg.bed_area is not None:
            x0, y0, x1, y1 = cfg.bed_area
            if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                raise ValueError("motion.bed_area must be [x0, y0, x1, y1] fractions with x0 < x1 and y0 < y1")
        if not 1 <= cfg.top_cells <= cfg.grid * cfg.grid:
            raise ValueError("motion.top_cells must be between 1 and grid*grid")
        return cfg


def bed_pixels(bed_area, width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bed_area if bed_area is not None else (0, 0, 1, 1)
    return int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height)


class BedFlow:
    """Frames -> per-cell motion vectors inside the bed area (uses OpenCV)."""

    def __init__(self, cfg: MotionConfig):
        import cv2
        self._cv2 = cv2
        self.cfg = cfg
        self._prev = None
        self._prev_t = None

    def process(self, frame_bgr, t: float):
        """Returns (grid*grid, 2) cell velocities in bed widths per second, or None for the first frame."""
        cv2 = self._cv2
        h, w = frame_bgr.shape[:2]
        x0, y0, x1, y1 = bed_pixels(self.cfg.bed_area, w, h)
        roi = frame_bgr[y0:y1, x0:x1]
        # Fixed analysis size: optical flow cost and noise level don't depend on camera resolution.
        scale = self.cfg.flow_width / roi.shape[1]
        size = (self.cfg.flow_width, max(self.cfg.grid, int(round(roi.shape[0] * scale))))
        gray = cv2.cvtColor(cv2.resize(roi, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)

        prev, prev_t = self._prev, self._prev_t
        self._prev, self._prev_t = gray, t
        if prev is None or prev.shape != gray.shape or t <= prev_t:
            return None
        flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 2, 9, 2, 5, 1.1, 0)
        g = self.cfg.grid
        ch, cw = gray.shape[0] // g, gray.shape[1] // g
        cells = flow[:ch * g, :cw * g].reshape(g, ch, g, cw, 2).mean(axis=(1, 3)).reshape(-1, 2)
        # pixels/frame -> bed widths/second, so thresholds hold at any resolution and frame rate
        return cells / gray.shape[1] / (t - prev_t)


@dataclass
class MotionEvent:
    start: float
    detected_at: float
    end: float | None = None
    confidence: float = 0.0

    @property
    def duration(self) -> float | None:
        return None if self.end is None else self.end - self.start


@dataclass
class MotionOutput:
    t: float
    intensity: float = 0.0           # mean speed of the most-moving cells, bed widths per second
    above: bool = False              # intensity above min_speed right now
    active_fraction: float = 0.0     # share of the last sustain_s that was above threshold
    building: bool = False           # intense movement seen, not yet sustained long enough
    current_event: MotionEvent | None = None
    event_started: MotionEvent | None = None
    event_ended: MotionEvent | None = None


class MotionDetector:
    """Pure logic: (t, cell velocities) -> sustained-movement events."""

    def __init__(self, cfg: MotionConfig, demo: bool = False):
        self.cfg = cfg
        self.sustain_s = cfg.demo_sustain_s if demo else cfg.sustain_s
        self._hist: deque = deque()        # (t, above_threshold)
        self._event: MotionEvent | None = None
        self._last_above: float | None = None
        self._run_start: float | None = None
        self._speeds: list[float] = []

    def update(self, t: float, cells) -> MotionOutput:
        cfg = self.cfg
        out = MotionOutput(t=t)
        if cells is not None:
            speeds = np.sort(np.linalg.norm(np.asarray(cells), axis=1))
            out.intensity = float(speeds[-cfg.top_cells:].mean())
        out.above = out.intensity >= cfg.min_speed

        self._hist.append((t, out.above))
        while self._hist and self._hist[0][0] < t - self.sustain_s:
            self._hist.popleft()
        span = t - self._hist[0][0]
        out.active_fraction = float(np.mean([a for _, a in self._hist]))

        if out.above:
            self._last_above = t
            if self._run_start is None:
                self._run_start = t
            self._speeds.append(out.intensity)
        elif self._event is None and self._last_above is not None and t - self._last_above > self.sustain_s:
            self._run_start, self._speeds = None, []

        if self._event is not None:
            self._event.confidence = self._confidence()
            if t - self._last_above >= cfg.end_hold_s:
                self._event.end = self._last_above
                out.event_ended = self._event
                self._event, self._run_start, self._speeds = None, None, []
        elif span >= self.sustain_s * 0.95 and out.active_fraction >= cfg.sustain_fraction:
            # "Most of the last sustain_s seconds" (not every frame): real movement has brief pauses.
            start = self._run_start if self._run_start is not None else t - self.sustain_s
            self._event = MotionEvent(start=start, detected_at=t, confidence=self._confidence())
            out.event_started = self._event

        out.current_event = self._event
        out.building = self._event is None and self._run_start is not None and out.active_fraction > 0
        return out

    def _confidence(self) -> float:
        """Heuristic 0-1, NOT a probability: how far movement exceeded the threshold, on average.
        0 = just at min_speed, 1 = at or above confidence_full_speed."""
        if not self._speeds:
            return 0.0
        lo, hi = self.cfg.min_speed, self.cfg.confidence_full_speed
        return float(np.clip((np.mean(self._speeds) - lo) / max(hi - lo, 1e-9), 0, 1))
