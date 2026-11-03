"""DIDS Phase 4A: EAR and MAR calculations; no camera or video loop."""
from __future__ import annotations
import numpy as np


def _points(points, minimum: int, name: str) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < minimum:
        raise ValueError(f"{name} must be an array shaped (at least {minimum}, 2)")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite coordinates")
    return arr


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(eye_points) -> float:
    """Calculate EAR from six ordered eye points, following Dlib eye order.

    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|).
    """
    p = _points(eye_points, 6, "eye_points")[:6]
    horizontal = _dist(p[0], p[3])
    if horizontal <= 1e-9:
        raise ValueError("Eye horizontal width is zero")
    return (_dist(p[1], p[5]) + _dist(p[2], p[4])) / (2.0 * horizontal)


def average_ear(left_eye, right_eye) -> float:
    """Return the mean EAR of the two eyes."""
    return (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0


def mouth_aspect_ratio(mouth_points) -> float:
    """Calculate a simple MAR from Dlib's 20 mouth points (indices 48..67).

    Uses three vertical mouth distances (landmark pairs 51-59, 52-58, 53-57)
    divided by twice the horizontal mouth width (48-54). This is a prototype
    geometry measure; tune its threshold with representative videos.
    """
    p = _points(mouth_points, 20, "mouth_points")[:20]
    horizontal = _dist(p[0], p[6])
    if horizontal <= 1e-9:
        raise ValueError("Mouth horizontal width is zero")
    vertical = _dist(p[3], p[9]) + _dist(p[4], p[8]) + _dist(p[5], p[7])
    return vertical / (3.0 * horizontal)
