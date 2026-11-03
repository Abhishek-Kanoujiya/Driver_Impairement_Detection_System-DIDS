"""DIDS Phase 4A: duration-based eye-closure and yawn event tracking."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import time
from collections import deque


@dataclass
class DrowsinessResult:
    ear: Optional[float]
    mar: Optional[float]
    eye_closed: bool
    prolonged_eye_closure: bool
    yawning: bool
    blink_event: bool
    yawn_event: bool


class DrowsinessAnalyzer:
    """Temporal EAR/MAR analyzer.

    Thresholds/durations are prototype defaults, not medically validated values.
    Pass timestamp=... for deterministic tests; otherwise monotonic time is used.
    """
    def __init__(self, ear_threshold: float = 0.23, mar_threshold: float = 0.65,
                 eye_closure_seconds: float = 1.0, yawn_seconds: float = 1.2,
                 smoothing_window: int = 5):
        if ear_threshold <= 0 or mar_threshold <= 0:
            raise ValueError("EAR and MAR thresholds must be positive")
        if eye_closure_seconds <= 0 or yawn_seconds <= 0:
            raise ValueError("Durations must be positive")
        if smoothing_window < 1:
            raise ValueError("smoothing_window must be at least 1")
        self.ear_threshold = float(ear_threshold)
        self.mar_threshold = float(mar_threshold)
        self.eye_closure_seconds = float(eye_closure_seconds)
        self.yawn_seconds = float(yawn_seconds)
        self._ears = deque(maxlen=int(smoothing_window))
        self._mars = deque(maxlen=int(smoothing_window))
        self._eye_closed_since: Optional[float] = None
        self._yawn_since: Optional[float] = None
        self._was_eye_closed = False
        self._was_yawning = False

    def update(self, ear: Optional[float], mar: Optional[float],
               timestamp: Optional[float] = None) -> DrowsinessResult:
        now = time.monotonic() if timestamp is None else float(timestamp)
        if ear is not None:
            if ear < 0:
                raise ValueError("EAR cannot be negative")
            self._ears.append(float(ear))
        if mar is not None:
            if mar < 0:
                raise ValueError("MAR cannot be negative")
            self._mars.append(float(mar))
        smooth_ear = sum(self._ears) / len(self._ears) if self._ears else None
        smooth_mar = sum(self._mars) / len(self._mars) if self._mars else None

        eye_closed = smooth_ear is not None and smooth_ear < self.ear_threshold
        yawning_now = smooth_mar is not None and smooth_mar > self.mar_threshold

        if eye_closed:
            if self._eye_closed_since is None:
                self._eye_closed_since = now
        else:
            self._eye_closed_since = None
        if yawning_now:
            if self._yawn_since is None:
                self._yawn_since = now
        else:
            self._yawn_since = None

        prolonged = (eye_closed and self._eye_closed_since is not None and
                     now - self._eye_closed_since >= self.eye_closure_seconds)
        sustained_yawn = (yawning_now and self._yawn_since is not None and
                          now - self._yawn_since >= self.yawn_seconds)
        blink_event = self._was_eye_closed and not eye_closed
        yawn_event = sustained_yawn and not self._was_yawning
        self._was_eye_closed = eye_closed
        self._was_yawning = sustained_yawn

        return DrowsinessResult(
            ear=smooth_ear, mar=smooth_mar, eye_closed=eye_closed,
            prolonged_eye_closure=prolonged, yawning=sustained_yawn,
            blink_event=blink_event, yawn_event=yawn_event
        )
