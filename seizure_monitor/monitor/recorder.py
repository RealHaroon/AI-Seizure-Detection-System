"""Event clip recording.

We keep the last `pre_event_s` seconds of frames in memory at all times. When
an event starts, those frames are written first, so the clip shows what
happened *before* detection (the detector needs several seconds of rhythm to
be sure). Recording then continues until `post_event_s` after the event ends.

Memory stays bounded: buffered frames are JPEG-compressed (~40 KB each
instead of ~900 KB raw), so 30 s at 15 fps is roughly 20 MB.

Writing happens in a background thread so encoding never slows down detection.
"""

from __future__ import annotations

import logging
import queue
import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from .overlay import draw_clip_banner, draw_skeleton

log = logging.getLogger(__name__)


class PrivacyFilter:
    """Pixelates the person in saved clips.

    Fail-safe: if the person can't be located (pose lost for longer than
    hold_s), the whole frame is pixelated rather than risk showing them.
    """

    def __init__(self, cfg: dict):
        self.enabled = cfg["blur_person"]
        self.blocks = cfg["pixel_blocks"]
        self.margin = cfg["margin"]
        self.hold_s = cfg["hold_s"]
        self._box = None          # last known person box (x0, y0, x1, y1)
        self._box_t = None

    def apply(self, frame, all_landmarks, t: float):
        if not self.enabled:
            return frame
        h, w = frame.shape[:2]
        if all_landmarks is not None:
            # Box around all 33 landmarks (including those MediaPipe estimates off-screen), plus a margin.
            xs = np.array([p[0] for p in all_landmarks])
            ys = np.array([p[1] for p in all_landmarks])
            mx, my = self.margin * (xs.max() - xs.min()), self.margin * (ys.max() - ys.min())
            self._box = (xs.min() - mx, ys.min() - my, xs.max() + mx, ys.max() + my)
            self._box_t = t
        if self._box is None or t - self._box_t > self.hold_s:
            x0, y0, x1, y1 = 0, 0, w, h
        else:
            x0, y0, x1, y1 = (int(np.clip(v, 0, lim)) for v, lim in zip(self._box, (w, h, w, h)))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return frame
        # Pixelation (shrink then enlarge) is cheaper than a big Gaussian blur and can't be "unblurred".
        # Block size scales with the person's size, so anonymisation is equally strong near or far.
        region = frame[y0:y1, x0:x1]
        block = max(x1 - x0, y1 - y0) / self.blocks
        small = cv2.resize(region, (max(1, round((x1 - x0) / block)), max(1, round((y1 - y0) / block))),
                           interpolation=cv2.INTER_AREA)
        frame[y0:y1, x0:x1] = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
        return frame


class _ClipWriter(threading.Thread):
    """Background thread: decodes buffered JPEGs and encodes them into an MP4.

    Frames arrive at an uneven rate (whatever the camera + pose estimation
    manage), but a video file has a fixed frame rate. We repeat or drop frames
    based on their timestamps so the clip plays back at real speed.
    """

    def __init__(self, path: Path, fps: float):
        super().__init__(daemon=True)
        self.path, self.fps = path, fps
        self.q: queue.Queue = queue.Queue()
        self._sink = None
        self._t0 = None
        self._written = 0

    def run(self):
        try:
            while (item := self.q.get()) is not None:
                t, jpeg = item
                frame = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
                if self._sink is None:
                    self._open(frame.shape[1], frame.shape[0])
                    self._t0 = t
                # Number of output frames that should exist up to time t at a fixed fps.
                target = int(round((t - self._t0) * self.fps)) + 1
                while self._written < target:
                    self._write(frame)
                    self._written += 1
        except Exception:
            log.exception("Clip writer failed for %s", self.path)
        finally:
            self._close()

    def _open(self, w, h):
        self._size = (w - w % 2, h - h % 2)   # H.264 needs even dimensions
        if shutil.which("ffmpeg"):
            # Browser-playable H.264 (needed for the dashboard). OpenCV's pip build can't encode H.264.
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                   "-s", f"{self._size[0]}x{self._size[1]}", "-r", str(self.fps), "-i", "-",
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
                   "-movflags", "+faststart", str(self.path)]
            self._sink = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            log.warning("ffmpeg not found: saving clip as MPEG-4 Part 2, which browsers can't play")
            self._sink = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self._size)

    def _write(self, frame):
        frame = frame[:self._size[1], :self._size[0]]
        if isinstance(self._sink, subprocess.Popen):
            self._sink.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            self._sink.write(frame)

    def _close(self):
        if isinstance(self._sink, subprocess.Popen):
            self._sink.stdin.close()
            self._sink.wait()
        elif self._sink is not None:
            self._sink.release()
        log.info("Clip saved: %s (%.1f s)", self.path, self._written / self.fps)


class EventRecorder:
    def __init__(self, rec_cfg: dict, privacy_cfg: dict, clips_dir: Path, source_name: str):
        self.pre_s = rec_cfg["pre_event_s"]
        self.post_s = rec_cfg["post_event_s"]
        self.fps = rec_cfg["fps"]
        self.jpeg_quality = rec_cfg["jpeg_quality"]
        self.clips_dir = Path(clips_dir)
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.source_name = source_name
        self.privacy = PrivacyFilter(privacy_cfg)

        self._buffer: deque = deque()   # (t, jpeg bytes), last pre_event_s seconds
        self._writer: _ClipWriter | None = None
        self._threads: list[_ClipWriter] = []   # includes clips still being flushed to disk
        self._clip_name: str | None = None
        self._stop_at: float | None = None   # source time at which to stop; None = event still running

    @property
    def recording(self) -> bool:
        return self._writer is not None

    @property
    def latest_jpeg(self) -> bytes | None:
        """Most recent clip frame (privacy filter + overlay applied), used as the alert snapshot."""
        return self._buffer[-1][1].tobytes() if self._buffer else None

    def add_frame(self, frame, pose_result, out, wall_time: float):
        """Call once per processed frame (after any start_event/end_event for this frame)."""
        t = out.t
        clip_frame = self.privacy.apply(frame.copy(), pose_result.all_landmarks, t)
        draw_skeleton(clip_frame, pose_result, out)   # drawn after pixelation so the movement stays visible
        draw_clip_banner(clip_frame, out, wall_time, self.source_name)
        ok, jpeg = cv2.imencode(".jpg", clip_frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return

        self._buffer.append((t, jpeg))
        while self._buffer and self._buffer[0][0] < t - self.pre_s:
            self._buffer.popleft()

        if self._writer is not None:
            self._writer.q.put((t, jpeg))
            if self._stop_at is not None and t >= self._stop_at:
                self._finish()

    def start_event(self, wall_time: float) -> str:
        """Begin (or extend) a recording. Returns the clip path relative to the data directory."""
        if self._writer is None:
            self._clip_name = time.strftime("event_%Y%m%d_%H%M%S.mp4", time.localtime(wall_time))
            self._writer = _ClipWriter(self.clips_dir / self._clip_name, self.fps)
            self._writer.start()
            self._threads = [th for th in self._threads if th.is_alive()] + [self._writer]
            for item in self._buffer:          # pre-event footage
                self._writer.q.put(item)
        # A new event during the post-event tail just keeps the same clip going.
        self._stop_at = None
        return f"{self.clips_dir.name}/{self._clip_name}"

    def end_event(self, event_end_t: float):
        self._stop_at = event_end_t + self.post_s

    def _finish(self):
        self._writer.q.put(None)
        self._writer = None
        self._stop_at = None

    def stop_now(self):
        """Monitoring paused: finish any clip immediately and drop the pre-event buffer
        (frames from before the pause shouldn't end up in the next event's clip)."""
        if self._writer is not None:
            self._finish()
        self._buffer.clear()

    def close(self):
        """Finish any clip in progress and wait for all clips to be written (call on shutdown)."""
        if self._writer is not None:
            self._finish()
        for th in self._threads:
            th.join(timeout=60)
        self._threads.clear()
