"""Video source abstraction: webcam index, video file, or RTSP URL.

Every frame comes with a timestamp in seconds. The detector works in real time
units, not frame counts, so timestamps must be right:
  - live sources (webcam/RTSP): wall-clock time when the frame was grabbed
  - video files: the video's own timeline (CAP_PROP_POS_MSEC), so a file gives
    identical results whether we play it in real time or process it faster
"""

from __future__ import annotations

import logging
import threading
import time

import cv2

log = logging.getLogger(__name__)


def parse_source(source: str):
    """'0' -> webcam 0; anything else is a file path or URL."""
    return int(source) if source.isdigit() else source


class VideoSource:
    def __init__(self, source: str, width: int, process_fps: float):
        self.source = parse_source(source)
        self.is_live = isinstance(self.source, int) or str(self.source).lower().startswith(("rtsp://", "http://", "https://"))
        self.width = width
        self.min_dt = 1.0 / process_fps

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        self._last_t: float | None = None
        if self.is_live:
            # Live cameras queue frames internally; if processing is slower than the camera we would
            # fall further and further behind. A grabber thread keeps only the newest frame instead.
            self._latest = None
            self._cond = threading.Condition()
            self._stopped = False
            self._thread = threading.Thread(target=self._grab_loop, daemon=True)
            self._thread.start()

    def _grab_loop(self):
        while not self._stopped:
            ok, frame = self.cap.read()
            t = time.monotonic()
            with self._cond:
                if not ok:
                    log.error("Live source stopped delivering frames")
                    self._stopped = True
                else:
                    self._latest = (frame, t)
                self._cond.notify_all()

    def read(self):
        """Return (frame_bgr_downscaled, t_seconds), or None when the source has ended."""
        if self.is_live:
            item = self._read_live()
        else:
            item = self._read_file()
        if item is None:
            return None
        frame, t = item
        self._last_t = t
        return self._downscale(frame), t

    def _read_live(self):
        deadline = time.monotonic() + 5.0
        with self._cond:
            while True:
                # Wait for a frame newer than the last one we used and at least min_dt later
                # (caps processing at process_fps to leave CPU for everything else).
                if self._latest is not None:
                    frame, t = self._latest
                    if self._last_t is None or t - self._last_t >= self.min_dt:
                        self._latest = None
                        return frame, t
                if self._stopped:
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    log.error("No frame from live source for 5 s")
                    return None
                self._cond.wait(timeout=min(remaining, self.min_dt))

    def _read_file(self):
        # Skip frames so we analyse at most process_fps, like we would live.
        while True:
            ok, frame = self.cap.read()
            if not ok:
                return None
            t = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if self._last_t is None or t - self._last_t >= self.min_dt * 0.999:
                return frame, t

    def _downscale(self, frame):
        h, w = frame.shape[:2]
        if w <= self.width:
            return frame
        return cv2.resize(frame, (self.width, int(round(h * self.width / w))), interpolation=cv2.INTER_AREA)

    def release(self):
        if self.is_live:
            self._stopped = True
            self._thread.join(timeout=1.0)
        self.cap.release()
