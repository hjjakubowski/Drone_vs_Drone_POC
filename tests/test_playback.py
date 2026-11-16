"""Regression checks for the two-screen demo and decoder lifecycle."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app.hud_desktop.main import _load_stylesheet
from app.hud_desktop.presenters.main_presenter import MainPresenter
from app.hud_desktop.state.view_definitions import load_view_definitions
from app.hud_desktop.ui.main_window import MainWindow
from app.hud_desktop.ui.views_page import ViewCard
from project.runtime.pipeline_runner import PipelineRuntime
from project.video.video_source import VideoSource


class PlaybackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setStyleSheet(_load_stylesheet())

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="inz-playback-test-")
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "sample.mp4"
        writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (96, 64))
        self.assertTrue(writer.isOpened())
        for index in range(4):
            writer.write(np.full((64, 96, 3), index * 60, dtype=np.uint8))
        writer.release()
        env = patch.dict(os.environ, {"INZ_RECORDINGS_DIR": self.directory.name})
        env.start()
        self.addCleanup(env.stop)

    def make_window(self):
        window = MainWindow()
        presenter = MainPresenter(window)
        self.addCleanup(window.close)
        window.show()
        self.app.processEvents()
        return window, presenter

    def test_source_and_pipeline_preserve_recording_time_until_eof(self):
        pipeline = PipelineRuntime(enable_analysis=False)
        self.addCleanup(pipeline.stop)
        pipeline.start(self.path)
        source = pipeline.source
        for index in range(1, 5):
            telemetry = pipeline.step()
            self.assertEqual(telemetry.frame_id, index)
            self.assertAlmostEqual(telemetry.dt, 1 / 25)
            self.assertEqual(telemetry.current_ms, index * 40)
            self.assertEqual(telemetry.duration_ms, 160)
            self.assertEqual(telemetry.progress_percent, index * 25)
            self.assertEqual(telemetry.frame.shape, (64, 96, 3))
        self.assertIsNone(pipeline.step())
        self.assertIsNone(pipeline.source)
        self.assertIsNone(source._capture)

    def test_reopening_and_failed_open_release_previous_recording(self):
        pipeline = PipelineRuntime(enable_analysis=False)
        self.addCleanup(pipeline.stop)
        pipeline.start(self.path)
        previous = pipeline.source
        pipeline.step()
        pipeline.start(self.path)
        self.assertIsNone(previous._capture)
        self.assertEqual(pipeline.step().frame_id, 1)
        with self.assertRaises(FileNotFoundError):
            pipeline.start(Path(self.directory.name) / "missing.mp4")
        self.assertIsNone(pipeline.source)
        broken = Path(self.directory.name) / "broken.mp4"
        broken.write_bytes(b"not a video")
        with self.assertRaises(OSError):
            VideoSource(broken)

    def test_four_cards_playback_back_and_escape(self):
        window, presenter = self.make_window()
        cards = window.views_page.findChildren(ViewCard)
        self.assertEqual(len(cards), 4)
        self.assertIs(window._stack.currentWidget(), window.views_page)
        with patch("app.hud_desktop.presenters.main_presenter.PipelineRuntime",
                   side_effect=lambda: PipelineRuntime(enable_analysis=False)):
            QTest.mouseClick(cards[0], Qt.MouseButton.LeftButton)
            self.assertIs(window._stack.currentWidget(), window.hud_page)
            for _ in range(100):
                QTest.qWait(2)
                if window.hud_page.video_canvas._frame is not None:
                    break
            self.assertIsNotNone(window.hud_page.video_canvas._frame)
            presenter._return_to_views()
            self.assertIs(window._stack.currentWidget(), window.views_page)
            presenter.shutdown()

    def test_eof_returns_to_selection_and_close_stops_playback(self):
        window, presenter = self.make_window()
        with patch("app.hud_desktop.presenters.main_presenter.PipelineRuntime",
                   side_effect=lambda: PipelineRuntime(enable_analysis=False)):
            presenter._start_hud("view-1")
            for _ in range(400):
                QTest.qWait(5)
                if window._stack.currentWidget() is window.views_page:
                    break
            self.assertIs(window._stack.currentWidget(), window.views_page)
            presenter._start_hud("view-2")
            window.close()
            self.assertEqual(presenter._workers, [])

    def test_missing_recordings_keep_four_placeholders_and_allow_return(self):
        self.path.unlink()
        definitions = load_view_definitions()
        self.assertEqual(len(definitions), 4)
        self.assertTrue(all(view.status == "NO VIDEO" for view in definitions))
        window, presenter = self.make_window()
        presenter._start_hud("view-4")
        for _ in range(100):
            QTest.qWait(2)
            if window.hud_page._playback_status.text() == "ERROR":
                break
        self.assertEqual(window.hud_page._playback_status.text(), "ERROR")
        self.assertIsNone(presenter._worker)
        QTest.keyClick(window, Qt.Key.Key_Escape)
        self.assertIs(window._stack.currentWidget(), window.views_page)


if __name__ == "__main__":
    unittest.main()
