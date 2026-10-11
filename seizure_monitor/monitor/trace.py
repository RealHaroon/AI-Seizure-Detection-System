"""Per-event movement trace, so a doctor can see WHY an alert fired.

Like the video recorder: we keep the last `pre_s` seconds of detector readings
in memory; when an event starts we keep recording until `post_s` after it
ends, then save one small JSON file per event:

    data/traces/event_<id>.json
    {"event_id": 7, "threshold": 0.04, "detected_at": 4.1, "end": 52.3,
     "samples": [[t, movement, rhythm_s], ...]}      # t = seconds from movement start

The dashboard draws it as a chart: movement over time, the alert level, and
the moment the alert fired.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path

log = logging.getLogger(__name__)


def trace_path(traces_dir: Path, event_id: int) -> Path:
    return Path(traces_dir) / f"event_{int(event_id)}.json"


class EventTrace:
    def __init__(self, traces_dir: Path, pre_s: float, post_s: float, threshold: float | None):
        self.dir = Path(traces_dir)
        self.pre_s, self.post_s, self.threshold = pre_s, post_s, threshold
        self._buffer: deque = deque()          # (t, movement, rhythm_s)
        self._current: dict | None = None      # event being captured
        self._stop_at: float | None = None

    @property
    def capturing(self) -> bool:
        return self._current is not None

    def add(self, t: float, movement: float | None, rhythm_s: float):
        sample = (t, movement, rhythm_s)
        self._buffer.append(sample)
        while self._buffer and self._buffer[0][0] < t - self.pre_s:
            self._buffer.popleft()
        if self._current is not None:
            self._current["samples"].append(sample)
            if self._stop_at is not None and t >= self._stop_at:
                self._write()

    def start(self, event_id: int, start: float, detected_at: float):
        if self._current is not None:
            self._write()                      # a new event during the previous one's tail: save the old one
        self._current = {"event_id": event_id, "start": start, "detected_at": detected_at, "end": None,
                         "samples": list(self._buffer)}
        self._stop_at = None

    def end(self, end: float):
        if self._current is not None:
            self._current["end"] = end
            self._stop_at = end + self.post_s

    def stop_now(self):
        """Monitoring paused or shutting down: save what we have."""
        if self._current is not None:
            self._write()
        self._buffer.clear()

    def _write(self):
        ev, self._current, self._stop_at = self._current, None, None
        t0 = ev["start"]
        data = {
            "event_id": ev["event_id"],
            "threshold": self.threshold,
            "detected_at": round(ev["detected_at"] - t0, 2),
            "end": None if ev["end"] is None else round(ev["end"] - t0, 2),
            "samples": [[round(t - t0, 2), None if m is None else round(m, 4), round(r, 1)]
                        for t, m, r in ev["samples"]],
        }
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            path = trace_path(self.dir, ev["event_id"])
            path.write_text(json.dumps(data, separators=(",", ":")))
            log.info("Movement trace saved: %s (%d samples)", path, len(data["samples"]))
        except OSError:
            log.exception("Could not save movement trace for event %s", ev["event_id"])
