"""DIDS Phase 2: reusable webcam frame preprocessing."""
from __future__ import annotations
import time
from collections import deque
from typing import Optional, Tuple
import cv2
import numpy as np

FrameSize = Optional[Tuple[int, int]]  # (width, height)

class FramePreprocessor:
    """Optionally resize, convert BGR to grayscale, then equalize histogram."""
    def __init__(self, target_size: FrameSize = (640, 480), equalize_histogram: bool = True):
        if target_size is not None:
            if (not isinstance(target_size, tuple) or len(target_size) != 2
                or not all(isinstance(v, int) and v > 0 for v in target_size)):
                raise ValueError("target_size must be None or a (width, height) tuple of positive integers.")
        self.target_size = target_size
        self.equalize_histogram = bool(equalize_histogram)

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """Return (processed uint8 grayscale frame, preprocessing time in ms)."""
        start = time.perf_counter()
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Input frame must be a non-empty NumPy array.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Input frame must be a BGR image with shape (height, width, 3).")
        if frame.dtype != np.uint8:
            raise ValueError("Input frame must use uint8 pixel values.")

        working = frame
        if self.target_size is not None:
            width, height = self.target_size
            if (frame.shape[1], frame.shape[0]) != (width, height):
                working = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        if self.equalize_histogram:
            gray = cv2.equalizeHist(gray)
        return gray, (time.perf_counter() - start) * 1000.0


def run_phase2(camera_index: int = 0, target_size: FrameSize = (640, 480),
               equalize_histogram: bool = True, requested_fps: int = 30) -> int:
    """Standalone webcam demonstration. FPS is measured, not assumed."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera index {camera_index}. Check permissions or try another index.")
        cap.release()
        return 1

    if target_size is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_size[1])
    if requested_fps > 0:
        cap.set(cv2.CAP_PROP_FPS, requested_fps)  # request only; hardware may ignore it

    processor = FramePreprocessor(target_size, equalize_histogram)
    timestamps = deque(maxlen=30)
    consecutive_failures = 0
    total_frames = 0
    total_processing_ms = 0.0
    print("Phase 2 running. Press Q or Esc to exit.")

    try:
        while True:
            success, frame = cap.read()
            if not success or frame is None or frame.size == 0:
                consecutive_failures += 1
                print(f"WARNING: frame read failed ({consecutive_failures}/30)")
                if consecutive_failures >= 30:
                    print("ERROR: 30 consecutive frame-read failures; stopping.")
                    return 1
                time.sleep(0.02)
                continue
            consecutive_failures = 0

            try:
                processed, processing_ms = processor.process(frame)
            except (ValueError, cv2.error) as exc:
                print(f"ERROR: preprocessing failed: {exc}")
                return 1

            total_frames += 1
            total_processing_ms += processing_ms
            timestamps.append(time.perf_counter())
            fps = 0.0
            if len(timestamps) >= 2:
                elapsed = timestamps[-1] - timestamps[0]
                if elapsed > 0:
                    fps = (len(timestamps) - 1) / elapsed
            average_ms = total_processing_ms / total_frames

            display = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            cv2.putText(display, f"FPS: {fps:.1f}", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(display, f"Preprocessing: {processing_ms:.2f} ms", (12, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display, f"Average: {average_ms:.2f} ms", (12, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2, cv2.LINE_AA)
            cv2.putText(display, "Q / ESC: Quit", (12, display.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("DIDS - Phase 2: Preprocessed Frame", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Frames processed: {total_frames}")
    if total_frames:
        print(f"Average preprocessing time: {total_processing_ms / total_frames:.2f} ms")
    print("Camera released.")
    return 0

if __name__ == "__main__":
    raise SystemExit(run_phase2())
