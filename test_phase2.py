
"""Offline tests for Phase 2; no webcam needed."""

import unittest
import cv2
import numpy as np
from phase2 import FramePreprocessor


class TestFramePreprocessor(unittest.TestCase):

    def setUp(self):
        self.frame = np.zeros((240, 320, 3), dtype=np.uint8)
        self.frame[:, :160] = (20, 80, 160)
        self.frame[:, 160:] = (220, 180, 40)

    def test_grayscale_output(self):
        out, elapsed = FramePreprocessor(target_size=None).process(self.frame)
        self.assertEqual(out.shape, (240, 320))
        self.assertEqual(out.dtype, np.uint8)
        self.assertGreaterEqual(elapsed, 0.0)

    def test_resize(self):
        out, _ = FramePreprocessor(target_size=(160, 120)).process(self.frame)
        self.assertEqual(out.shape, (120, 160))

    def test_equalization_can_be_disabled(self):
        out, _ = FramePreprocessor(
            target_size=None,
            equalize_histogram=False
        ).process(self.frame)

        expected = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        np.testing.assert_array_equal(out, expected)

    def test_empty_frame_rejected(self):
        with self.assertRaises(ValueError):
            FramePreprocessor().process(np.array([], dtype=np.uint8))

    def test_grayscale_input_rejected(self):
        with self.assertRaises(ValueError):
            FramePreprocessor().process(np.zeros((100, 100), dtype=np.uint8))

    def test_non_uint8_frame_rejected(self):
        invalid_frame = np.zeros((100, 100, 3), dtype=np.float32)

        with self.assertRaises(ValueError):
            FramePreprocessor().process(invalid_frame)


if __name__ == "__main__":
    unittest.main(verbosity=2)
