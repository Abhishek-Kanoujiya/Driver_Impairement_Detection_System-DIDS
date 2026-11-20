import unittest
import cv2
import numpy as np

import head_pose
from head_pose import (
    face_model_3d_points,
    facial_landmark_indices,
    extract_image_points,
    get_camera_matrix,
    get_distortion_coefficients,
    estimate_head_pose,
    decompose_rotation_vector,
    draw_nose_direction,
    draw_head_pose_info,
    HeadPoseTracker,
)
from face_landmarks import FaceLandmarks


class TestHeadPoseEstimation(unittest.TestCase):
    def setUp(self):
        self.frame_width = 640
        self.frame_height = 480
        self.camera_matrix = get_camera_matrix(self.frame_width, self.frame_height)
        self.dist_coeffs = get_distortion_coefficients()

    def test_face_model_points_shape_and_indices(self):
        self.assertEqual(face_model_3d_points.shape, (6, 3))
        self.assertEqual(len(facial_landmark_indices), 6)
        self.assertEqual(facial_landmark_indices, [30, 8, 36, 45, 48, 54])

    def test_camera_matrix_values(self):
        self.assertEqual(self.camera_matrix.shape, (3, 3))
        self.assertEqual(self.camera_matrix[0, 0], 640.0)
        self.assertEqual(self.camera_matrix[1, 1], 640.0)
        self.assertEqual(self.camera_matrix[0, 2], 320.0)
        self.assertEqual(self.camera_matrix[1, 2], 240.0)
        self.assertEqual(self.camera_matrix[2, 2], 1.0)

    def test_extract_image_points_from_array_and_facelandmarks(self):
        raw_points = np.zeros((68, 2), dtype=np.int32)
        for i in range(68):
            raw_points[i] = [i * 2, i * 3]

        extracted = extract_image_points(raw_points)
        self.assertEqual(extracted.shape, (6, 2))
        self.assertEqual(extracted[0, 0], raw_points[30, 0])
        self.assertEqual(extracted[1, 1], raw_points[8, 1])

        face = FaceLandmarks((0, 0, 100, 100), raw_points)
        extracted_from_face = extract_image_points(face)
        np.testing.assert_array_equal(extracted, extracted_from_face)

    def test_extract_image_points_invalid(self):
        with self.assertRaises(ValueError):
            extract_image_points(np.zeros((67, 2)))

    def test_frontal_pose_estimation_zero_angles(self):
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.array([[0.0], [0.0], [1000.0]], dtype=np.float64)

        proj, _ = cv2.projectPoints(
            face_model_3d_points,
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs
        )
        image_2d_points = proj.reshape(-1, 2)

        pose = estimate_head_pose(image_2d_points, self.camera_matrix, self.dist_coeffs)
        self.assertIsNotNone(pose)
        pitch_deg, yaw_deg, roll_deg, rotation_vec, translation_vec = pose

        self.assertAlmostEqual(pitch_deg, 0.0, places=1)
        self.assertAlmostEqual(yaw_deg, 0.0, places=1)
        self.assertAlmostEqual(roll_deg, 0.0, places=1)

    def test_pitch_and_yaw_rotation_recovery(self):
        tvec = np.array([[0.0], [0.0], [1000.0]], dtype=np.float64)

        # 18 degrees pitch down
        target_pitch = 18.0
        rvec_pitch = np.array([[np.deg2rad(target_pitch)], [0.0], [0.0]], dtype=np.float64)
        proj_pitch, _ = cv2.projectPoints(
            face_model_3d_points,
            rvec_pitch,
            tvec,
            self.camera_matrix,
            self.dist_coeffs
        )
        pitch_deg, _, _, _, _ = estimate_head_pose(
            proj_pitch.reshape(-1, 2),
            self.camera_matrix,
            self.dist_coeffs
        )
        self.assertAlmostEqual(pitch_deg, target_pitch, delta=1.5)

        # 22 degrees yaw turned away
        target_yaw = 22.0
        rvec_yaw = np.array([[0.0], [np.deg2rad(target_yaw)], [0.0]], dtype=np.float64)
        proj_yaw, _ = cv2.projectPoints(
            face_model_3d_points,
            rvec_yaw,
            tvec,
            self.camera_matrix,
            self.dist_coeffs
        )
        _, yaw_deg, _, _, _ = estimate_head_pose(
            proj_yaw.reshape(-1, 2),
            self.camera_matrix,
            self.dist_coeffs
        )
        self.assertAlmostEqual(yaw_deg, target_yaw, delta=1.5)


class TestHeadPoseTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = HeadPoseTracker(
            pitch_threshold=15.0,
            yaw_threshold=20.0,
            nodding_consecutive_frames=3,
            distraction_consecutive_frames=3
        )
        self.cam = get_camera_matrix(640, 480)
        self.dist = get_distortion_coefficients()
        self.rvec = np.zeros((3, 1))
        self.tvec = np.zeros((3, 1))

    def test_output_dictionary_keys(self):
        result = self.tracker.update(0.0, 0.0, 0.0, self.rvec, self.tvec, self.cam, self.dist)
        required_keys = [
            "pitch_deg", "yaw_deg", "roll_deg",
            "is_nodding", "is_distracted",
            "rotation_vec", "translation_vec",
            "camera_matrix", "dist_coeffs"
        ]
        for key in required_keys:
            self.assertIn(key, result)

    def test_consecutive_frame_nodding_detection(self):
        # Frame 1: pitch exceeded
        res1 = self.tracker.update(18.0, 0.0, 0.0, self.rvec, self.tvec, self.cam, self.dist)
        self.assertFalse(res1["is_nodding"])
        self.assertEqual(res1["nodding_frame_count"], 1)

        # Frame 2: pitch exceeded
        res2 = self.tracker.update(19.0, 0.0, 0.0, self.rvec, self.tvec, self.cam, self.dist)
        self.assertFalse(res2["is_nodding"])
        self.assertEqual(res2["nodding_frame_count"], 2)

        # Frame 3: threshold reached -> is_nodding = True
        res3 = self.tracker.update(20.0, 0.0, 0.0, self.rvec, self.tvec, self.cam, self.dist)
        self.assertTrue(res3["is_nodding"])
        self.assertEqual(res3["nodding_frame_count"], 3)

        # Normal frame resets count
        res_reset = self.tracker.update(5.0, 0.0, 0.0, self.rvec, self.tvec, self.cam, self.dist)
        self.assertFalse(res_reset["is_nodding"])
        self.assertEqual(res_reset["nodding_frame_count"], 0)

    def test_consecutive_frame_distraction_detection(self):
        # 3 consecutive frames with yaw > 20 deg
        self.tracker.update(0.0, 25.0, 0.0, self.rvec, self.tvec, self.cam, self.dist)
        self.tracker.update(0.0, 24.0, 0.0, self.rvec, self.tvec, self.cam, self.dist)
        res = self.tracker.update(0.0, 26.0, 0.0, self.rvec, self.tvec, self.cam, self.dist)

        self.assertTrue(res["is_distracted"])
        self.assertFalse(res["is_nodding"])
        self.assertEqual(res["distraction_frame_count"], 3)

    def test_missing_face_handling(self):
        self.tracker.update(0.0, 25.0, 0.0, self.rvec, self.tvec, self.cam, self.dist)
        missing = self.tracker.handle_missing_face()

        self.assertIsNone(missing["pitch_deg"])
        self.assertIsNone(missing["yaw_deg"])
        self.assertFalse(missing["is_nodding"])
        self.assertFalse(missing["is_distracted"])
        self.assertEqual(missing["distraction_frame_count"], 0)


class TestOverlayDrawing(unittest.TestCase):
    def test_draw_nose_direction_runs(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cam = get_camera_matrix(640, 480)
        dist = get_distortion_coefficients()
        rvec = np.zeros((3, 1))
        tvec = np.array([[0.0], [0.0], [1000.0]])
        proj, _ = cv2.projectPoints(face_model_3d_points, rvec, tvec, cam, dist)
        image_points = proj.reshape(-1, 2)

        out = draw_nose_direction(frame, image_points, rvec, tvec, cam, dist)
        self.assertEqual(out.shape, (480, 640, 3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
