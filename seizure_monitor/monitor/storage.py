"""SQLite storage for detected events (the "seizure diary").

An event row is inserted as soon as an event starts (end_time still NULL), so
the event is on record even if the program crashes or the power goes out
mid-event. The row is completed when the event ends.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

REVIEW_LABELS = ("unreviewed", "real_seizure", "false_alarm")

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    start_time    TEXT NOT NULL,          -- ISO 8601 local time when rhythmic movement began
    end_time      TEXT,                   -- NULL while the event is still in progress
    duration_s    REAL,
    confidence    REAL,                   -- heuristic 0-1 strength score, not a probability
    clip_path     TEXT,                   -- relative to the data directory
    source        TEXT NOT NULL,          -- camera name from config
    review_label  TEXT NOT NULL DEFAULT 'unreviewed'
                  CHECK (review_label IN {REVIEW_LABELS}),
    reviewer_note TEXT NOT NULL DEFAULT ''
);
"""


class EventStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            # WAL lets the dashboard read while the monitor writes, without "database is locked" errors.
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)

    def _connect(self):
        # A short-lived connection per call keeps this safe to use from any thread.
        db = sqlite3.connect(self.db_path, timeout=5)
        db.row_factory = sqlite3.Row
        return db

    def start_event(self, start_time: str, confidence: float, clip_path: str | None, source: str) -> int:
        with closing(self._connect()) as db, db:
            cur = db.execute(
                "INSERT INTO events (start_time, confidence, clip_path, source) VALUES (?, ?, ?, ?)",
                (start_time, confidence, clip_path, source))
            return cur.lastrowid

    def end_event(self, event_id: int, end_time: str, duration_s: float, confidence: float):
        with closing(self._connect()) as db, db:
            db.execute("UPDATE events SET end_time = ?, duration_s = ?, confidence = ? WHERE id = ?",
                       (end_time, duration_s, confidence, event_id))

    def list_events(self) -> list[dict]:
        with closing(self._connect()) as db:
            return [dict(r) for r in db.execute("SELECT * FROM events ORDER BY start_time DESC")]

    def get_event(self, event_id: int) -> dict | None:
        with closing(self._connect()) as db:
            row = db.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
            return dict(row) if row else None

    def set_review(self, event_id: int, label: str, note: str = ""):
        if label not in REVIEW_LABELS:
            raise ValueError(f"review label must be one of {REVIEW_LABELS}")
        with closing(self._connect()) as db, db:
            db.execute("UPDATE events SET review_label = ?, reviewer_note = ? WHERE id = ?",
                       (label, note, event_id))
