"""Checking pipeline and communication with UI, without using GPU."""

import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app.hud_desktop.presenters.main_presenter import MainPresenter
from app.hud_desktop.ui.main_window import MainWindow
from project.runtime.pipeline_runner import PipelineRuntime


class FakeSource:
    last = None

    def __init__(self, path):
        FakeSource.last = self
        self.i = 0
        self.fps = 30.0
        self.dt = 1 / 30
        self.duration_ms = 100
        self.fps_is_estimated = True
        self.closed = False

    def read_frame(self):
        self.i += 1
        if self.i > 3:
            return False, None, self.dt, self.i
        return True, np.zeros((1080, 1920, 3), dtype=np.uint8), self.dt, self.i

    def close(self):
        self.closed = True


class FakeTracker:
    last = None

    def __init__(self, *args, **kwargs):
        FakeTracker.last = self
        self.i = 0
        self.reset_called = False

    def process_frame(self, frame):
        self.i += 1
        return SimpleNamespace(target_id=1, detected=self.i != 3, missing_frames=int(self.i == 3),
                               state=np.array([960., 540., 120. + self.i, 50., 0, 0, 1, 0]))

    def reset(self):
        self.reset_called = True


class PipelineIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_runtime_uses_source_time_and_clears_state(self):
        with patch("project.runtime.pipeline_runner.VideoSource", FakeSource), \
             patch("project.vision.tracker_engine.VisionTracker", FakeTracker):
            pipeline = PipelineRuntime(enable_analysis=True)
            pipeline.start("example.mp4")
            a, b, c = pipeline.step(), pipeline.step(), pipeline.step()
            self.assertIsNone(a.navigation.closing_speed)
            self.assertIsNotNone(b.navigation.closing_speed)
            self.assertFalse(c.tracking.detected)
            self.assertIsNone(c.navigation.depth_z)
            self.assertTrue(a.fps_is_estimated)
            self.assertAlmostEqual(a.dt, 1 / 30)
            self.assertGreater(b.analysis_fps, 0)
            self.assertIsNone(pipeline.step())
            self.assertTrue(FakeSource.last.closed)
            self.assertTrue(FakeTracker.last.reset_called)
            self.assertIsNone(pipeline.estimator)

    def test_open_failure_releases_video_source(self):
        with patch("project.runtime.pipeline_runner.VideoSource", FakeSource), \
             patch("project.vision.tracker_engine.VisionTracker", side_effect=OSError("model missing")):
            pipeline = PipelineRuntime(enable_analysis=True)
            with self.assertRaisesRegex(OSError, "model missing"):
                pipeline.start("example.mp4")
            self.assertTrue(FakeSource.last.closed)
            self.assertIsNone(pipeline.source)
            self.assertIsNone(pipeline.estimator)

    def test_wrong_resolution_is_reported_and_worker_releases_source(self):
        class SmallSource(FakeSource):
            def read_frame(self):
                self.i += 1
                return True, np.zeros((540, 960, 3), dtype=np.uint8), self.dt, self.i

        with patch("project.runtime.pipeline_runner.VideoSource", SmallSource), \
             patch("project.vision.tracker_engine.VisionTracker", FakeTracker):
            window = MainWindow()
            presenter = MainPresenter(window)
            try:
                window.show()
                presenter._start_video("invalid-size.mp4")
                for _ in range(300):
                    QTest.qWait(5)
                    if window.hud_page._playback_status.text() == "ERROR":
                        break
                self.assertEqual(window.hud_page._playback_status.text(), "ERROR")
                self.assertIn("1920", window.hud_page._playback_status.toolTip())
                self.assertTrue(FakeSource.last.closed)
            finally:
                presenter.shutdown()
                window.close()

    def test_worker_gui_displays_results_and_finishes(self):
        with patch("project.runtime.pipeline_runner.VideoSource", FakeSource), \
             patch("project.vision.tracker_engine.VisionTracker", FakeTracker):
            window = MainWindow()
            presenter = MainPresenter(window)
            self.addCleanup(presenter.shutdown)
            self.addCleanup(window.close)
            received = []
            original = window.hud_page.show_frame

            def observe(telemetry):
                received.append((telemetry.frame_id, threading.current_thread(), telemetry))
                original(telemetry)
                window.hud_page.video_canvas.repaint()

            window.hud_page.show_frame = observe
            window.show()
            presenter._start_hud("view-1")
            for _ in range(300):
                QTest.qWait(5)
                if len(received) == 3 and presenter._worker is None:
                    break
            self.assertEqual([item[0] for item in received], [1, 2, 3])
            self.assertTrue(all(item[1] is threading.main_thread() for item in received))
            self.assertIsNotNone(received[1][2].navigation.closing_speed)
            self.assertFalse(received[2][2].tracking.detected)
            self.assertEqual(window.hud_page.telemetry_panel._values["target"].text(), "PREDICTION ID 1")
            self.assertEqual(window.hud_page.telemetry_panel._values["depth"].text(), "--")
            self.assertIn("est.", window.hud_page.telemetry_panel._values["source_fps"].text())
            self.assertTrue(FakeSource.last.closed)
            self.assertIs(window._stack.currentWidget(), window.views_page)

    def test_return_while_inference_is_running_and_open_another_video(self):
        entered = threading.Event()
        release = threading.Event()

        class SlowTracker(FakeTracker):
            def process_frame(self, frame):
                entered.set()
                release.wait(3)
                return super().process_frame(frame)

        with patch("project.runtime.pipeline_runner.VideoSource", FakeSource), \
             patch("project.vision.tracker_engine.VisionTracker", SlowTracker):
            window = MainWindow()
            presenter = MainPresenter(window)
            try:
                window.show()
                presenter._start_video("first.mp4")
                for _ in range(300):
                    QTest.qWait(5)
                    if entered.is_set():
                        break
                self.assertTrue(entered.is_set())
                old_worker = presenter._worker
                presenter._return_to_views()
                self.assertIs(window._stack.currentWidget(), window.views_page)
                self.assertTrue(old_worker.isRunning())
                presenter._start_video("second.mp4")
                self.assertIsNotNone(presenter._pending_view)
                release.set()
                for _ in range(300):
                    QTest.qWait(5)
                    if presenter._pending_view is None and presenter._worker is not old_worker:
                        break
                self.assertIsNone(presenter._pending_view)
                self.assertIsNot(presenter._worker, old_worker)
            finally:
                release.set()
                presenter.shutdown()
                window.close()


if __name__ == "__main__":
    unittest.main()
