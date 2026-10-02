"""Rhythm-based convulsive movement detector.

How it works, in one paragraph: MediaPipe gives us body keypoints per frame.
For each tracked point we look at its trajectory over a sliding window and ask
"is this point oscillating rhythmically at a frequency typical of clonic
jerking (default 1.5-6 Hz)?" using an FFT. If several limbs oscillate at once,
and this persists for several seconds, we declare a seizure *event*.

This is signal processing on top of AI pose estimation. It is NOT a trained
seizure classifier, and it can only see convulsive (motor) seizures.

This module is pure logic: no camera, no OpenCV, no MediaPipe. Input is
timestamped keypoints, output is state + features, so it can be unit-tested
with synthetic signals.
"""

from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass, field, fields, replace

import numpy as np
from scipy.signal import detrend

# MediaPipe-style names of the points we track (shoulders also give us scale).
TRACKED_POINTS = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]
_IDX = {name: i for i, name in enumerate(TRACKED_POINTS)}

# Prevents division by zero when a spectrum is (almost) perfectly flat or empty.
_EPS = 1e-12


class State(enum.Enum):
    NORMAL = "NORMAL"
    RHYTHMIC = "RHYTHMIC"            # rhythm seen, but not yet long enough to be an event
    SEIZURE_EVENT = "SEIZURE EVENT"


@dataclass
class DetectorConfig:
    resample_hz: float
    window_s: float
    hop_s: float
    min_visibility: float
    max_gap_s: float
    body_lost_reset_s: float
    min_shoulder_width_px: float
    band_hz: tuple
    min_amplitude: float
    min_band_ratio: float
    min_prominence: float
    limbs: dict
    min_rhythmic_limbs: int
    onset_s: float
    demo_onset_s: float
    rhythm_gap_tolerance_s: float
    end_hold_s: float
    confidence: dict

    @classmethod
    def from_dict(cls, d: dict) -> "DetectorConfig":
        # Fail loudly on typos or missing keys in config.yaml instead of silently using wrong values.
        names = {f.name for f in fields(cls)}
        unknown, missing = set(d) - names, names - set(d)
        if unknown or missing:
            raise ValueError(f"detector config: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
        cfg = cls(**d)
        cfg.band_hz = tuple(cfg.band_hz)
        for limb, pts in cfg.limbs.items():
            bad = [p for p in pts if p not in _IDX]
            if bad:
                raise ValueError(f"detector config: limb {limb!r} has unknown points {bad}")
        return cfg


@dataclass
class PointFeatures:
    amplitude: float      # RMS displacement over the window, in shoulder widths
    band_ratio: float     # fraction of movement power inside the band (0-1)
    prominence: float     # in-band peak power / median in-band power
    peak_hz: float        # frequency of the strongest in-band component
    rhythmic: bool


@dataclass
class Event:
    start: float                 # time of the first rhythmic window of the run
    detected_at: float           # time the event was declared (start + onset delay)
    end: float | None = None     # time of the last rhythmic window
    confidence: float = 0.0

    @property
    def duration(self) -> float | None:
        return None if self.end is None else self.end - self.start


@dataclass
class DetectorOutput:
    t: float
    state: State
    body_visible: bool
    warming_up: bool                       # not enough history yet to analyse a full window
    window_rhythmic: bool
    points: dict = field(default_factory=dict)        # point name -> PointFeatures
    rhythmic_limbs: list = field(default_factory=list)
    visible_limbs: list = field(default_factory=list)
    window_confidence: float = 0.0
    rhythm_duration_s: float = 0.0         # how long the current rhythm run has lasted
    onset_s: float = 0.0                   # how long it must last to become an event
    current_event: Event | None = None
    event_started: Event | None = None     # set only on the update where an event begins
    event_ended: Event | None = None       # set only on the update where an event ends


class Detector:
    def __init__(self, cfg: DetectorConfig, demo: bool = False):
        self.cfg = cfg
        self.onset_s = cfg.demo_onset_s if demo else cfg.onset_s
        self.n_window = int(round(cfg.window_s * cfg.resample_hz))
        # Keep slightly more raw history than one window so the window start can be interpolated.
        self._history_s = cfg.window_s + cfg.max_gap_s + 1.0
        self.reset()

        # Precompute the FFT frequency axis and the Hann window (same for every point and window).
        self._freqs = np.fft.rfftfreq(self.n_window, d=1.0 / cfg.resample_hz)
        self._hann = np.hanning(self.n_window)
        lo, hi = cfg.band_hz
        self._band = (self._freqs >= lo) & (self._freqs <= hi)
        if hi > cfg.resample_hz / 2:
            raise ValueError("band_hz upper edge must be below resample_hz / 2 (Nyquist)")

    def reset(self):
        """Forget everything (used at start and if the body is lost for too long)."""
        self._times: deque[float] = deque()
        self._obs: deque[np.ndarray] = deque()   # each: (n_points, 2) pixel xy, NaN where missing
        self._last_t: float | None = None
        self._last_shoulders_t: float | None = None
        self._last_eval_t: float | None = None
        self._last_out: DetectorOutput | None = None
        # State machine
        self._run_start: float | None = None
        self._last_rhythmic_t: float | None = None
        self._run_confidences: list[float] = []
        self._event: Event | None = None

    # ------------------------------------------------------------------ input

    def update(self, t: float, keypoints: dict | None) -> DetectorOutput:
        """Feed one frame.

        t:          timestamp in seconds (must increase; real time, not frame count)
        keypoints:  {point_name: (x_px, y_px, visibility)} or None if no person found
        """
        if self._last_t is not None and t <= self._last_t:
            # Out-of-order or duplicate timestamp: ignore rather than corrupt the time series.
            return self._repeat_last(self._last_t) if self._last_out else self._empty_output(t)
        self._last_t = t

        xy = np.full((len(TRACKED_POINTS), 2), np.nan)
        if keypoints:
            for name, (x, y, vis) in keypoints.items():
                i = _IDX.get(name)
                if i is not None and vis >= self.cfg.min_visibility:
                    xy[i] = (x, y)

        shoulders_ok = not np.isnan(xy[[_IDX["left_shoulder"], _IDX["right_shoulder"]]]).any()
        if shoulders_ok:
            self._last_shoulders_t = t
        elif self._last_shoulders_t is not None and t - self._last_shoulders_t > self.cfg.body_lost_reset_s:
            # Body lost for too long: old samples would be stitched to new ones across a big gap and
            # produce garbage spectra. Drop the history (but keep event state so the event can end cleanly).
            self._times.clear()
            self._obs.clear()
            self._last_shoulders_t = None

        self._times.append(t)
        self._obs.append(xy)
        while self._times and self._times[0] < t - self._history_s:
            self._times.popleft()
            self._obs.popleft()

        # Re-analyse only every hop_s; between hops repeat the last result (cheap, and stable on screen).
        if self._last_eval_t is not None and t - self._last_eval_t < self.cfg.hop_s:
            out = self._repeat_last(t)
            out.body_visible = shoulders_ok
            return out
        self._last_eval_t = t

        out = self._analyse_window(t)
        out.body_visible = shoulders_ok
        self._step_state_machine(out)
        self._last_out = out
        return out

    # --------------------------------------------------------------- features

    def _analyse_window(self, t: float) -> DetectorOutput:
        cfg = self.cfg
        out = self._empty_output(t)
        times = np.fromiter(self._times, float)
        if len(times) < 2 or times[0] > t - cfg.window_s:
            out.warming_up = True
            return out

        obs = np.stack(self._obs)                                  # (n_frames, n_points, 2)
        grid = t - cfg.window_s + np.arange(self.n_window) / cfg.resample_hz

        # Scale: median shoulder width in this window, so movement is in "shoulder widths"
        # and doesn't depend on how far the person is from the camera.
        in_win = times >= grid[0]
        sw = np.linalg.norm(obs[in_win, _IDX["left_shoulder"]] - obs[in_win, _IDX["right_shoulder"]], axis=1)
        sw = sw[~np.isnan(sw)]
        if len(sw) == 0 or np.median(sw) < cfg.min_shoulder_width_px:
            return out
        scale = float(np.median(sw))

        for name, i in _IDX.items():
            series = self._resample_point(times, obs[:, i], grid)
            if series is not None:
                out.points[name] = self._point_features(series / scale)

        # A limb counts as visible if any of its points produced features, rhythmic if any point is rhythmic.
        for limb, pts in cfg.limbs.items():
            feats = [out.points[p] for p in pts if p in out.points]
            if feats:
                out.visible_limbs.append(limb)
                if any(f.rhythmic for f in feats):
                    out.rhythmic_limbs.append(limb)

        out.window_rhythmic = len(out.rhythmic_limbs) >= cfg.min_rhythmic_limbs
        out.window_confidence = self._window_confidence(out)
        return out

    def _resample_point(self, times, xy, grid):
        """Interpolate one point's trajectory onto the uniform grid.

        Returns (n_window, 2) or None if the point has a gap longer than max_gap_s
        anywhere in the window (including at the window edges): filling a long gap
        with a straight line would invent movement that never happened.
        """
        valid = ~np.isnan(xy[:, 0])
        tv = times[valid]
        if len(tv) < 2:
            return None
        # Walk from the window start to the window end through the real observations; every step
        # (including from the last observation to "now") must be at most max_gap_s.
        before = tv[tv <= grid[0]]
        start = before[-1] if len(before) else grid[0]
        inside = tv[(tv > grid[0]) & (tv <= grid[-1])]
        if np.max(np.diff(np.concatenate(([start], inside, [grid[-1]])))) > self.cfg.max_gap_s:
            return None
        return np.column_stack([np.interp(grid, tv, xy[valid, 0]), np.interp(grid, tv, xy[valid, 1])])

    def _point_features(self, series: np.ndarray) -> PointFeatures:
        cfg = self.cfg
        # Detrend: remove slow drift (e.g. the person slowly sliding) so it doesn't dominate the spectrum.
        d = detrend(series, axis=0, type="linear")
        amplitude = float(np.sqrt(np.mean(np.sum(d ** 2, axis=1))))

        # Hann window reduces spectral leakage from cutting the signal at the window edges.
        spec = np.abs(np.fft.rfft(d * self._hann[:, None], axis=0)) ** 2
        power = spec.sum(axis=1)            # x + y power: movement in any direction counts
        power[0] = 0.0                      # ignore DC (should be ~0 after detrending anyway)

        band_power = power[self._band]
        band_ratio = float(band_power.sum() / (power.sum() + _EPS))
        k = int(np.argmax(band_power))
        prominence = float(band_power[k] / (np.median(band_power) + _EPS))
        peak_hz = float(self._freqs[self._band][k])

        rhythmic = (amplitude >= cfg.min_amplitude and band_ratio >= cfg.min_band_ratio
                    and prominence >= cfg.min_prominence)
        return PointFeatures(amplitude, band_ratio, prominence, peak_hz, rhythmic)

    def _window_confidence(self, out: DetectorOutput) -> float:
        """Heuristic 0-1 strength score for one window. NOT a probability of seizure.

            confidence = w_limbs * (rhythmic limbs / visible limbs)
                       + w_band  * mean band score over rhythmic points
                       + w_prom  * mean prominence score over rhythmic points

        band score       = (band_ratio - min_band_ratio) / (band_ratio_full - min_band_ratio), clipped to 0-1
        prominence score = same idea using min_prominence and prominence_full

        So 0 means "barely passes / no rhythm" and 1 means "rhythm in every visible
        limb, almost all power in the band, and a very sharp peak".
        """
        c = self.cfg.confidence
        rhythmic_pts = [f for f in out.points.values() if f.rhythmic]
        if not rhythmic_pts or not out.visible_limbs:
            return 0.0
        limb_frac = len(out.rhythmic_limbs) / len(out.visible_limbs)
        band = np.mean([_ramp(f.band_ratio, self.cfg.min_band_ratio, c["band_ratio_full"]) for f in rhythmic_pts])
        prom = np.mean([_ramp(f.prominence, self.cfg.min_prominence, c["prominence_full"]) for f in rhythmic_pts])
        total = c["weight_limbs"] + c["weight_band"] + c["weight_prominence"]
        return float((c["weight_limbs"] * limb_frac + c["weight_band"] * band + c["weight_prominence"] * prom) / total)

    # ---------------------------------------------------------- state machine

    def _step_state_machine(self, out: DetectorOutput):
        cfg, t = self.cfg, out.t

        if out.window_rhythmic:
            if self._run_start is None:
                self._run_start = t
                self._run_confidences = []
            self._last_rhythmic_t = t
            self._run_confidences.append(out.window_confidence)
        elif (self._event is None and self._run_start is not None
              and t - self._last_rhythmic_t > cfg.rhythm_gap_tolerance_s):
            # Rhythm stopped before reaching onset_s: it was a short burst (e.g. waving), not an event.
            self._run_start = None

        if self._event is not None:
            self._event.confidence = float(np.mean(self._run_confidences))
            # Hysteresis: only end after rhythm has been absent for end_hold_s, so a brief pause
            # in the jerking doesn't split one seizure into several alerts.
            if t - self._last_rhythmic_t >= cfg.end_hold_s:
                self._event.end = self._last_rhythmic_t
                out.event_ended = self._event
                self._event = None
                self._run_start = None
        elif self._run_start is not None and self._last_rhythmic_t - self._run_start >= self.onset_s:
            self._event = Event(start=self._run_start, detected_at=t,
                                confidence=float(np.mean(self._run_confidences)))
            out.event_started = self._event

        out.current_event = self._event
        out.onset_s = self.onset_s
        out.rhythm_duration_s = 0.0 if self._run_start is None else self._last_rhythmic_t - self._run_start
        if self._event is not None:
            out.state = State.SEIZURE_EVENT
        elif self._run_start is not None:
            out.state = State.RHYTHMIC
        else:
            out.state = State.NORMAL

    def _repeat_last(self, t: float) -> DetectorOutput:
        # Copy so the start/end notification is delivered exactly once.
        return replace(self._last_out, t=t, event_started=None, event_ended=None)

    def _empty_output(self, t: float) -> DetectorOutput:
        return DetectorOutput(t=t, state=State.NORMAL, body_visible=False, warming_up=False,
                              window_rhythmic=False, onset_s=self.onset_s)


def _ramp(x: float, lo: float, hi: float) -> float:
    """Linear 0-1 score: 0 at lo, 1 at hi, clipped."""
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))
