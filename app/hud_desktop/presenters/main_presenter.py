from __future__ import annotations

import cv2
from PySide6.QtCore import QObject, QTimer

from app.hud_desktop.state.view_definitions import load_view_definitions
from app.hud_desktop.ui.main_window import MainWindow
from project.runtime.pipeline_runner import PipelineRuntime


class MainPresenter(QObject):
    def __init__(self, window: MainWindow) -> None:
        super().__init__(window)
        self._window = window
        self._views = {view.id: view for view in load_view_definitions()}
        self.runtime = PipelineRuntime()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._read_frame)

        self._window.views_page.set_views(list(self._views.values()))
        self._window.views_page.view_selected.connect(self._start_hud)
        self._window.hud_page.back_requested.connect(self._return_to_views)
        self._window.closing.connect(self.stop)

    def _start_hud(self, view_id: str) -> None:
        selected_view = self._views.get(view_id)
        if selected_view is None:
            self._window.show_error("Selected view not found.")
            return

        self.stop()
        self._window.hud_page.start_view(selected_view)
        self._window.show_hud()
        try:
            self.runtime.start(selected_view.video_path)
        except (OSError, cv2.error) as error:
            self._window.hud_page.show_error(str(error))
            return
        self._timer.start(max(1, round(1000 / self.runtime.source.fps)))
        self._read_frame()

    def _read_frame(self) -> None:
        try:
            telemetry = self.runtime.step()
        except (OSError, cv2.error) as error:
            self.stop()
            self._window.hud_page.show_error(str(error))
            return
        if telemetry is None:
            self._return_to_views()
        else:
            self._window.hud_page.show_frame(telemetry)

    def stop(self) -> None:
        self._timer.stop()
        self.runtime.stop()

    def _return_to_views(self) -> None:
        self.stop()
        self._window.show_views()
