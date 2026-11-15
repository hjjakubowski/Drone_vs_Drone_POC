"""Target-lock logic tests without GPU or inference."""
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from project.config import load_config, project_root
from project.vision.tracker_engine import VisionTracker


class VisionTrackerTests(unittest.TestCase):
    def setUp(self):
        self.model = MagicMock()
        self.backend = SimpleNamespace(tracked_stracks=[], lost_stracks=[], reset=MagicMock())
        self.model.predictor.trackers = [self.backend]
        patcher = patch("project.vision.tracker_engine.YOLO", return_value=self.model)
        patcher.start()
        self.addCleanup(patcher.stop)
        config = load_config("model")["model"]
        self.tracker = VisionTracker(project_root() / config["weights"], max_missing_frames=2)
        self.frame = np.zeros((8, 8, 3), dtype=np.uint8)

    def step(self, ids=(), tracked=(), lost=()):
        self.backend.tracked_stracks = list(tracked)
        self.backend.lost_stracks = list(lost)
        boxes = MagicMock()
        boxes.id.int.return_value.cpu.return_value.tolist.return_value = list(ids)
        self.model.track.return_value = [SimpleNamespace(boxes=boxes)]
        return self.tracker.process_frame(self.frame)

    def make_track(self, track_id):
        return SimpleNamespace(track_id=track_id, mean=np.array([10., 20., 5., 4., 1., 2., 0., 0.]), is_activated=True)

    def test_no_detection_and_first_target(self):
        self.assertIsNone(self.step().state)
        a, b = self.make_track(7), self.make_track(8)
        result = self.step([7, 8], [a, b])
        self.assertEqual(result.target_id, 7)
        self.assertTrue(result.detected)
        self.assertFalse(np.shares_memory(result.state, a.mean))
        self.assertEqual(self.model.track.call_args.kwargs["conf"], 0.1)

    def test_prediction_limit_and_reacquisition(self):
        a = self.make_track(7)
        self.step([7], [a])
        for n in (1, 2):
            result = self.step(lost=[a])
            self.assertEqual(result.target_id, 7)
            self.assertFalse(result.detected)
            self.assertEqual(result.missing_frames, n)
            np.testing.assert_array_equal(result.state, a.mean)
        result = self.step(lost=[a])
        self.assertIsNone(result.target_id)
        self.assertIsNone(result.state)
        b = self.make_track(8)
        self.assertEqual(self.step([8], [b]).target_id, 8)

    def test_return_after_loss_and_do_not_switch_to_other_target(self):
        a, b = self.make_track(7), self.make_track(8)
        self.step([7], [a])
        result = self.step([8], [b], [a])
        self.assertEqual(result.target_id, 7)
        self.assertFalse(result.detected)
        result = self.step([7, 8], [a, b])
        self.assertEqual(result.target_id, 7)
        self.assertTrue(result.detected)
        self.assertEqual(result.missing_frames, 0)

    def test_missing_track_and_reset(self):
        a = self.make_track(7)
        self.step([7], [a])
        result = self.step()
        self.assertIsNone(result.state)
        self.assertEqual(result.target_id, 7)
        self.tracker.reset()
        self.backend.reset.assert_called_once()
        self.assertIsNone(self.tracker.active_target_id)
        self.assertEqual(self.tracker.missing_frames, 0)

    def test_missing_boxes_id(self):
        self.model.track.return_value = [SimpleNamespace(boxes=SimpleNamespace(id=None))]
        self.assertIsNone(self.tracker.process_frame(self.frame).state)


if __name__ == "__main__":
    unittest.main()
