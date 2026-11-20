import argparse
from collections import deque
from pathlib import Path
import time
import cv2
import numpy as np

from phase2 import FramePreprocessor
from face_landmarks import FaceLandmarkDetector, draw_landmarks
from ear_mar import average_ear, mouth_aspect_ratio
from drowsiness import DrowsinessAnalyzer
from head_pose import (
    HeadPoseTracker,
    extract_image_points,
    get_camera_matrix,
    get_distortion_coefficients,
    estimate_head_pose,
    draw_nose_direction,
    draw_head_pose_info
)


def run_pipeline(camera_index: int = 0,
                 model_path: str = "models/shape_predictor_68_face_landmarks.dat",
                 pitch_threshold: float = 15.0,
                 yaw_threshold: float = 20.0,
                 target_size: tuple[int, int] = (640, 480)) -> int:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_size[1])

    try:
        landmark_detector = FaceLandmarkDetector(model_path)
    except (ImportError, FileNotFoundError) as error:
        print(f"Error initializing landmark detector: {error}")
        cap.release()
        return 1

    preprocessor = FramePreprocessor(target_size=target_size)
    drowsiness_analyzer = DrowsinessAnalyzer()
    head_pose_tracker = HeadPoseTracker(
        pitch_threshold=pitch_threshold,
        yaw_threshold=yaw_threshold
    )

    timestamps = deque(maxlen=30)
    print("DIDS pipeline active. Press Q or Esc to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                print("Warning: Failed to capture camera frame.")
                time.sleep(0.01)
                continue

            timestamps.append(time.perf_counter())
            fps = 0.0
            if len(timestamps) >= 2:
                elapsed = timestamps[-1] - timestamps[0]
                if elapsed > 0:
                    fps = (len(timestamps) - 1) / elapsed

            gray, preprocessing_ms = preprocessor.process(frame)
            display_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            faces = landmark_detector.detect(gray)

            if faces:
                face = faces[0]
                display_frame = draw_landmarks(display_frame, [face])

                ear = average_ear(face.left_eye, face.right_eye)
                mar = mouth_aspect_ratio(face.mouth)
                drowsiness_result = drowsiness_analyzer.update(ear, mar)

                image_2d_points = extract_image_points(face.points)
                camera_matrix = get_camera_matrix(target_size[0], target_size[1])
                dist_coeffs = get_distortion_coefficients()

                pose_estimate = estimate_head_pose(image_2d_points, camera_matrix, dist_coeffs)
                if pose_estimate is not None:
                    pitch_deg, yaw_deg, roll_deg, rotation_vec, translation_vec = pose_estimate
                    pose_result = head_pose_tracker.update(
                        pitch_deg=pitch_deg,
                        yaw_deg=yaw_deg,
                        roll_deg=roll_deg,
                        rotation_vec=rotation_vec,
                        translation_vec=translation_vec,
                        camera_matrix=camera_matrix,
                        dist_coeffs=dist_coeffs
                    )

                    draw_nose_direction(
                        display_frame,
                        image_2d_points,
                        rotation_vec,
                        translation_vec,
                        camera_matrix,
                        dist_coeffs
                    )
                    draw_head_pose_info(display_frame, pose_result)

                cv2.putText(display_frame, f"EAR: {ear:.2f}", (12, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(display_frame, f"MAR: {mar:.2f}", (12, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

                if drowsiness_result.prolonged_eye_closure:
                    cv2.putText(display_frame, "DROWSINESS ALERT: EYES CLOSED", (12, 280),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
                elif drowsiness_result.yawning:
                    cv2.putText(display_frame, "FATIGUE ALERT: YAWNING DETECTED", (12, 310),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2, cv2.LINE_AA)
            else:
                head_pose_tracker.handle_missing_face()
                drowsiness_analyzer.update(None, None)
                cv2.putText(display_frame, "NO FACE DETECTED", (12, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.putText(display_frame, f"FPS: {fps:.1f}", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(display_frame, "Q / ESC: Quit", (12, target_size[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow("Driver Impairment Detection System (DIDS)", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    except KeyboardInterrupt:
        print("Pipeline interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released. Pipeline shut down cleanly.")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Driver Impairment Detection System (DIDS)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--model", type=str,
                        default="models/shape_predictor_68_face_landmarks.dat",
                        help="Path to shape_predictor_68_face_landmarks.dat")
    parser.add_argument("--nod-pitch", type=float, default=15.0,
                        help="Pitch threshold in degrees for head nodding (default: 15.0)")
    parser.add_argument("--distract-yaw", type=float, default=20.0,
                        help="Yaw threshold in degrees for distraction (default: 20.0)")
    parser.add_argument("--width", type=int, default=640, help="Target frame width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Target frame height (default: 480)")
    args = parser.parse_args()

    return run_pipeline(
        camera_index=args.camera,
        model_path=args.model,
        pitch_threshold=args.nod_pitch,
        yaw_threshold=args.distract_yaw,
        target_size=(args.width, args.height)
    )


if __name__ == "__main__":
    raise SystemExit(main())
