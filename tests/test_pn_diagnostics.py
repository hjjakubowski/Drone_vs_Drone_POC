"""Porównanie okien i zapis pojedynczej próbki bez modelu i GPU."""

import unittest
from types import SimpleNamespace

import numpy as np

from project.config import load_config
from scripts.test_pn_estimator import SmoothingComparison, csv_record


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.compare = SmoothingComparison(load_config("navigation")["navigation"], 10)

    @staticmethod
    def track(width=100, detected=True, target_id=1):
        return SimpleNamespace(target_id=target_id, detected=detected, missing_frames=int(not detected),
                               state=np.array([960., 540., width, 50., 0., 0., 0., 0.]))

    def step(self, frame, track):
        return self.compare.process(track, 0.1, frame, (1920, 1080), False)

    def test_full_windows_shared_frames_and_reset(self):
        self.assertEqual(self.compare.windows, {"baseline": 3, "100ms": 1, "200ms": 2, "300ms": 3})
        for i in range(1, 6):
            results, ready = self.step(i, self.track(width=100 + i))
        self.assertTrue(all(ready.values()))
        results, ready = self.step(6, self.track(detected=False))
        self.assertFalse(any(ready.values()))
        self.assertIsNone(results["baseline"].closing_speed)
        results, ready = self.step(7, self.track())
        self.assertFalse(any(ready.values()))
        results, ready = self.step(8, self.track(target_id=2))
        self.assertFalse(any(ready.values()))
        self.assertIsNone(results["baseline"].closing_speed)

    def test_csv_tracks_dimensions_deltas_status_and_unavailable_values(self):
        first = self.track()
        results, ready = self.step(1, first)
        row1 = csv_record(1, 0.1, first, results, ready, None)
        self.assertEqual(row1["width_px"], 100)
        self.assertIsNone(row1["delta_width_px"])
        self.assertIsNone(row1["closing_speed"])
        second = self.track(width=102)
        results, ready = self.step(2, second)
        row2 = csv_record(2, 0.1, second, results, ready, row1)
        self.assertEqual(row2["delta_width_px"], 2)
        prediction = self.track(detected=False)
        results, ready = self.step(3, prediction)
        row3 = csv_record(3, 0.1, prediction, results, ready, row2)
        self.assertEqual(row3["status"], "prediction")
        self.assertIsNone(row3["depth_z"])
        self.assertIsNone(row3["delta_width_px"])


if __name__ == "__main__":
    unittest.main()
