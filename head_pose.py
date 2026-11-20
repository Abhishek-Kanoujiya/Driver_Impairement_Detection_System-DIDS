import cv2
import numpy as np

face_model_3d_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),         # Chin
    (-225.0, 170.0, -135.0),      # Left eye outer corner
    (225.0, 170.0, -135.0),       # Right eye outer corner
    (-150.0, -150.0, -125.0),     # Left mouth corner
    (150.0, -150.0, -125.0)       # Right mouth corner
], dtype=np.float64)

facial_landmark_indices = [30, 8, 36, 45, 48, 54]


def extract_image_points(landmarks):
    if hasattr(landmarks, "points"):
        points = landmarks.points
    else:
        points = np.asarray(landmarks)

    if points.shape[0] < 68 or points.shape[1] != 2:
        raise ValueError("Landmarks array must contain at least 68 (x, y) points")

    image_2d_points = np.array([points[idx] for idx in facial_landmark_indices], dtype=np.float64)
    return image_2d_points


def get_camera_matrix(frame_width, frame_height):
    focal_length = float(frame_width)
    center_x = frame_width / 2.0
    center_y = frame_height / 2.0

    camera_matrix = np.array([
        [focal_length, 0.0, center_x],
        [0.0, focal_length, center_y],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    return camera_matrix


def get_distortion_coefficients():
    return np.zeros((4, 1), dtype=np.float64)
