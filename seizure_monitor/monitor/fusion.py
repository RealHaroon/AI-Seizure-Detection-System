"""Combine the two detectors into one stream of events.

  - Detector A (motion.py): sustained intense movement in the bed area.
  - Detector B (detector.py): rhythmic jerking in several limbs (pose-based).

An event starts as soon as EITHER detector starts one and ends only when BOTH
are quiet, so alerts, clips and the event log see a single event per episode.
The event records which detector(s) triggered it, so a doctor can see whether
it was "rhythmic convulsion", "sustained vigorous movement", or both.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .detector import DetectorOutput, State
from .motion import MotionOutput

TRIGGER_TEXT = {
    "rhythmic": "rhythmic convulsive movement",
    "movement": "sustained vigorous movement",
    "rhythmic+movement": "rhythmic convulsive movement and sustained vigorous movement",
}


@dataclass
class FusedEvent:
    start: float
    detected_at: float
    triggers: set = field(default_factory=set)   # {"rhythmic", "movement"}
    end: float | None = None
    confidence: float = 0.0

    @property
    def trigger(self) -> str:
        return "+".join(t for t in ("rhythmic", "movement") if t in self.triggers)

    @property
    def duration(self) -> float | None:
        return None if self.end is None else self.end - self.start


@dataclass
class FusedOutput:
    t: float
    state: State
    pose: DetectorOutput
    motion: MotionOutput | None
    current_event: FusedEvent | None = None
    event_started: FusedEvent | None = None
    event_ended: FusedEvent | None = None

    @property
    def points(self):
        # Lets the skeleton drawing colour points by rhythm, exactly as before.
        return self.pose.points


class EventFuser:
    def __init__(self):
        self._event: FusedEvent | None = None
        self._last_end = None

    def update(self, pose: DetectorOutput, motion: MotionOutput | None) -> FusedOutput:
        t = pose.t
        pose_ev = pose.current_event
        mot_ev = motion.current_event if motion else None
        out = FusedOutput(t=t, state=State.NORMAL, pose=pose, motion=motion)

        # Track when each detector's own event finished; the fused event ends at the latest of these.
        for done in (pose.event_ended, motion.event_ended if motion else None):
            if done is not None and done.end is not None:
                self._last_end = done.end if self._last_end is None else max(self._last_end, done.end)

        if pose_ev is not None or mot_ev is not None:
            if self._event is None:
                starts = [e.start for e in (pose_ev, mot_ev) if e is not None]
                self._event = FusedEvent(start=min(starts), detected_at=t)
                out.event_started = self._event
            ev = self._event
            if pose_ev is not None:
                ev.triggers.add("rhythmic")
            if mot_ev is not None:
                ev.triggers.add("movement")
            ev.confidence = max([ev.confidence] + [e.confidence for e in (pose_ev, mot_ev) if e is not None])
        elif self._event is not None:
            ev = self._event
            ev.end = max(self._last_end if self._last_end is not None else t, ev.start)
            out.event_ended = ev
            self._event, self._last_end = None, None

        if self._event is None:
            self._last_end = None
        out.current_event = self._event
        if self._event is not None:
            out.state = State.SEIZURE_EVENT
        elif pose.state == State.RHYTHMIC:
            out.state = State.RHYTHMIC
        elif motion is not None and motion.building:
            out.state = State.MOVEMENT
        return out
