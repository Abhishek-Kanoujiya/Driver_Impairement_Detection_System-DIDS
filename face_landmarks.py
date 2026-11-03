"""DIDS Phase 3: reusable Dlib 68-point facial landmark localization.

This module accepts an already-processed grayscale frame. It never opens a camera.
Dlib's 68-point convention (zero-based Python indices):
  jaw 0:17, right eyebrow 17:22, left eyebrow 22:27, nose 27:36,
  right eye 36:42, left eye 42:48, outer/inner mouth 48:68.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

try:
    import dlib
except ImportError:  # Keep offline EAR/MAR tests usable without Dlib installed.
    dlib = None


@dataclass
class FaceLandmarks:
    """Landmarks for one detected face; coordinates are (x, y) in input-frame pixels."""
    rect: tuple[int, int, int, int]  # left, top, right, bottom
    points: np.ndarray               # shape (68, 2), dtype int32

    @property
    def right_eye(self) -> np.ndarray:
        return self.points[36:42]

    @property
    def left_eye(self) -> np.ndarray:
        return self.points[42:48]

    @property
    def mouth(self) -> np.ndarray:
        return self.points[48:68]


class FaceLandmarkDetector:
    """Dlib frontal-face detector plus the 68-point predictor."""

    def __init__(self, predictor_path: str | Path = "models/shape_predictor_68_face_landmarks.dat",
                 upsample: int = 0):
        if dlib is None:
            raise ImportError("Dlib is required for Phase 3. Install dlib/dlib-bin compatible with your Python.")
        path = Path(predictor_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"68-point predictor model not found: {path}. Download "
                "shape_predictor_68_face_landmarks.dat and place it at this path."
            )
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(str(path))
        self.upsample = int(upsample)

    def detect(self, gray: np.ndarray) -> list[FaceLandmarks]:
        """Return zero or more faces. Raises ValueError for invalid frame input."""
        if not isinstance(gray, np.ndarray) or gray.size == 0:
            raise ValueError("gray must be a non-empty NumPy array")
        if gray.ndim != 2 or gray.dtype != np.uint8:
            raise ValueError("gray must be a 2-D uint8 grayscale frame")
        rects = self.detector(gray, self.upsample)
        faces: list[FaceLandmarks] = []
        for rect in rects:
            try:
                shape = self.predictor(gray, rect)
                points = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)],
                                  dtype=np.int32)
                faces.append(FaceLandmarks(
                    (rect.left(), rect.top(), rect.right(), rect.bottom()), points
                ))
            except (RuntimeError, ValueError, IndexError):
                # A failed predictor on one face should not crash the full frame.
                continue
        return faces


def draw_landmarks(frame: np.ndarray, faces: list[FaceLandmarks],
                   draw_indices: bool = False) -> np.ndarray:
    """Draw face boxes and landmarks on a BGR or grayscale image; returns a copy."""
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("frame must be a non-empty NumPy array")
    out = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
    for face in faces:
        x1, y1, x2, y2 = face.rect
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
        for i, (x, y) in enumerate(face.points):
            cv2.circle(out, (int(x), int(y)), 2, (0, 255, 255), -1)
            if draw_indices:
                cv2.putText(out, str(i), (int(x) + 2, int(y) - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1, cv2.LINE_AA)
    return out
