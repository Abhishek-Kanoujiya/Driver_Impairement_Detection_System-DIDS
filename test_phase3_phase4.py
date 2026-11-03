"""Offline tests for Phase 3 coordinate contract and Phase 4A math/state logic."""
import unittest
import numpy as np
from ear_mar import eye_aspect_ratio, average_ear, mouth_aspect_ratio
from drowsiness import DrowsinessAnalyzer
from face_landmarks import FaceLandmarks, draw_landmarks


class TestEARMAR(unittest.TestCase):
    def test_open_eye_ear_is_positive(self):
        eye = np.array([[0, 0], [2, -2], [6, -2], [8, 0], [6, 2], [2, 2]], float)
        self.assertGreater(eye_aspect_ratio(eye), 0.0)

    def test_closed_eye_ear_is_lower(self):
        open_eye = np.array([[0, 0], [2, -2], [6, -2], [8, 0], [6, 2], [2, 2]], float)
        closed_eye = np.array([[0, 0], [2, -.1], [6, -.1], [8, 0], [6, .1], [2, .1]], float)
        self.assertGreater(eye_aspect_ratio(open_eye), eye_aspect_ratio(closed_eye))

    def test_average_ear(self):
        eye = np.array([[0, 0], [2, -2], [6, -2], [8, 0], [6, 2], [2, 2]], float)
        self.assertAlmostEqual(average_ear(eye, eye), eye_aspect_ratio(eye))

    def test_invalid_eye_input(self):
        with self.assertRaises(ValueError):
            eye_aspect_ratio([[0, 0], [1, 1]])
        with self.assertRaises(ValueError):
            eye_aspect_ratio(np.full((6, 2), np.nan))

    def test_zero_width_eye_rejected(self):
        with self.assertRaises(ValueError):
            eye_aspect_ratio(np.zeros((6, 2)))

    def test_mouth_ratio(self):
        mouth = np.array([
            [0,0],[1,-1],[3,-2],[5,-2],[7,-1],[9,0],[10,0],
            [9,1],[7,2],[5,2],[3,2],[1,1],
            [1,0],[3,0],[5,0],[7,0],[9,0],[7,1],[5,1],[3,1]
        ], float)
        self.assertGreaterEqual(mouth_aspect_ratio(mouth), 0)

    def test_invalid_mouth_rejected(self):
        with self.assertRaises(ValueError):
            mouth_aspect_ratio(np.zeros((10, 2)))


class TestDrowsiness(unittest.TestCase):
    def test_prolonged_closure_uses_duration(self):
        a = DrowsinessAnalyzer(ear_threshold=.2, eye_closure_seconds=1,
                               smoothing_window=1)
        self.assertFalse(a.update(.1, .1, timestamp=0).prolonged_eye_closure)
        self.assertFalse(a.update(.1, .1, timestamp=.7).prolonged_eye_closure)
        self.assertTrue(a.update(.1, .1, timestamp=1.1).prolonged_eye_closure)

    def test_short_blink_does_not_trigger_prolonged(self):
        a = DrowsinessAnalyzer(ear_threshold=.2, eye_closure_seconds=1,
                               smoothing_window=1)
        a.update(.1, .1, timestamp=0)
        self.assertFalse(a.update(.1, .1, timestamp=.2).prolonged_eye_closure)
        result = a.update(.4, .1, timestamp=.3)
        self.assertFalse(result.prolonged_eye_closure)
        self.assertTrue(result.blink_event)

    def test_sustained_yawn(self):
        a = DrowsinessAnalyzer(mar_threshold=.5, yawn_seconds=.8, smoothing_window=1)
        self.assertFalse(a.update(.3, .8, timestamp=0).yawning)
        self.assertFalse(a.update(.3, .8, timestamp=.5).yawning)
        self.assertTrue(a.update(.3, .8, timestamp=.9).yawning)

    def test_missing_measurements_do_not_crash(self):
        a = DrowsinessAnalyzer()
        result = a.update(None, None, timestamp=1)
        self.assertIsNone(result.ear)
        self.assertIsNone(result.mar)


class TestLandmarkDrawing(unittest.TestCase):
    def test_draw_helper_and_coordinate_contract(self):
        pts = np.zeros((68, 2), dtype=np.int32)
        face = FaceLandmarks((0, 0, 20, 20), pts)
        result = draw_landmarks(np.zeros((30, 30), dtype=np.uint8), [face])
        self.assertEqual(result.shape, (30, 30, 3))
        self.assertEqual(face.right_eye.shape, (6, 2))
        self.assertEqual(face.left_eye.shape, (6, 2))
        self.assertEqual(face.mouth.shape, (20, 2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
