"""Testy estymatora nawigacji bez modelu, GPU i interfejsu graficznego."""

import math
import unittest
from types import SimpleNamespace

import numpy as np

from project.config import load_config
from project.navigation.pn_estimator import ProportionalNavigation


def tracking(x=960, y=540, w=279.165, h=50, vw=0, vh=0, target_id=1, detected=True):
    return SimpleNamespace(target_id=target_id, detected=detected,
                           state=np.array([x, y, w, h, 0, 0, vw, vh], dtype=float))


class NavigationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config("navigation")["navigation"]
        self.estimator = ProportionalNavigation(self.config)

    def process(self, frame, target=None, dt=0.1, image_size=(1920, 1080), estimated=False):
        return self.estimator.process(target or tracking(), dt, frame, image_size, estimated)

    def test_first_detection_angles_depth_and_unavailable_derivatives(self):
        result = self.process(1, estimated=True)
        self.assertEqual((result.los_x, result.los_y), (0, 0))
        self.assertAlmostEqual(result.depth_z, 1.0)
        self.assertTrue(result.fps_is_estimated)
        for field in ("los_rate_x", "los_rate_y", "closing_speed", "ttc", "accel_x", "accel_y"):
            self.assertIsNone(getattr(result, field))
        self.estimator.reset()
        off_center = self.process(1, tracking(x=1060, y=640))
        self.assertGreater(off_center.los_x, 0)
        self.assertGreater(off_center.los_y, 0)

    def test_depth_rate_rolling_mean_and_guidance(self):
        k = self.config["fx"] * self.config["target_width_m"]
        self.process(1, tracking(w=k / 4))
        second = self.process(2, tracking(x=1060, y=640, w=k / 3), dt=1)
        self.assertAlmostEqual(second.closing_speed, 1)
        self.assertAlmostEqual(second.accel_x, 3 * second.los_rate_x)
        self.assertGreater(second.accel_y, 0)
        third = self.process(3, tracking(w=k / 1), dt=1)
        self.assertAlmostEqual(third.closing_speed, 1.5)
        fourth = self.process(4, tracking(w=k / 2), dt=1)
        self.assertAlmostEqual(fourth.closing_speed, (1 + 2 - 1) / 3)
        fifth = self.process(5, tracking(w=k / 3), dt=1)
        self.assertAlmostEqual(fifth.closing_speed, 0)
        self.assertIsNone(fifth.accel_x)

    def test_static_target_and_ttc_units_and_threshold(self):
        self.process(1, tracking(w=100, h=50, vw=2, vh=1))
        result = self.process(2, tracking(w=100, h=50, vw=2, vh=1))
        self.assertAlmostEqual(result.ttc, 5)
        self.assertEqual(result.closing_speed, 0)
        self.assertIsNone(result.accel_x)
        self.assertIsNone(self.process(3, tracking(w=100, h=50)).ttc)
        self.assertIsNone(self.process(4, tracking(w=100, h=50, vh=-1)).ttc)
        self.estimator.ttc_denominator_min = 1000
        self.assertIsNone(self.process(5, tracking(w=100, h=50, vh=1)).ttc)

    def test_loss_prediction_invalid_state_and_target_change_clear_history(self):
        for invalid in (tracking(detected=False), tracking(target_id=None),
                        SimpleNamespace(target_id=1, detected=True, state=None),
                        SimpleNamespace(target_id=1, detected=True, state=[1, 2]),
                        tracking(w=0), tracking(w=float("nan"))):
            with self.subTest(invalid=invalid):
                self.estimator.reset()
                self.process(1)
                empty = self.process(2, invalid)
                self.assertIsNone(empty.depth_z)
                self.assertIsNone(self.process(3).closing_speed)
        self.estimator.reset()
        self.process(1)
        self.assertIsNone(self.process(2, tracking(target_id=2)).closing_speed)
        self.assertIsNone(self.process(4, tracking(target_id=2)).closing_speed)
        self.estimator.reset()
        self.assertIsNone(self.process(5).closing_speed)

    def test_invalid_input_clears_history(self):
        for dt, size in ((0, (1920, 1080)), (math.nan, (1920, 1080)),
                         (-1, (1920, 1080)), (0.1, (960, 540)), (0.1, None)):
            with self.subTest(dt=dt, size=size):
                self.estimator.reset()
                self.process(1)
                with self.assertRaises(ValueError):
                    self.process(2, dt=dt, image_size=size)
                self.assertIsNone(self.process(3).closing_speed)

    def test_invalid_config(self):
        for key, value in (("fx", 0), ("fy", float("nan")), ("cx", None),
                           ("target_width_m", -1), ("guidance_gain", -1),
                           ("ttc_denominator_min", -1), ("closing_speed_window", 0),
                           ("image_width", 1920.0)):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    ProportionalNavigation({**self.config, key: value})


if __name__ == "__main__":
    unittest.main()
