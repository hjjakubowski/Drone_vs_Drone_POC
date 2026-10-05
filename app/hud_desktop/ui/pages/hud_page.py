from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from app.hud_desktop.state.view_definitions import ViewDefinition
from app.hud_desktop.ui.widgets.status_strip import StatusStrip
from app.hud_desktop.ui.widgets.telemetry_panel import TelemetryPanel
from app.hud_desktop.ui.widgets.video_hud_canvas import VideoHudCanvas
from app.hud_desktop.utils.time_format import format_ms


class HudPage(QWidget):
    hud_finished = Signal()
    back_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("hudPage")

        self._active_view: ViewDefinition | None = None

        self._view_name = QLabel("WIDOK")
        self._view_name.setObjectName("hudTitle")
        self._playback_status = QLabel("IDLE")
        self._playback_status.setObjectName("statusReady")
        self._time_label = QLabel("00:00 / 00:00")
        self._time_label.setObjectName("hudTime")

        self._back_button = QPushButton("Powrot")
        self._back_button.setObjectName("secondaryButton")
        self._back_button.clicked.connect(self.request_back)

        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(18, 12, 18, 12)
        top_bar.setSpacing(18)
        top_bar.addWidget(self._view_name)
        top_bar.addWidget(self._playback_status)
        top_bar.addStretch(1)
        top_bar.addWidget(self._time_label)
        top_bar.addWidget(self._back_button)

        top_frame = QFrame()
        top_frame.setObjectName("hudTopBar")
        top_frame.setLayout(top_bar)

        self.video_canvas = VideoHudCanvas()
        self.video_canvas.finished.connect(self._on_video_finished)
        self.video_canvas.playback_changed.connect(self._on_playback_changed)

        self.status_strip = StatusStrip()
        self.telemetry_panel = TelemetryPanel()

        right_panel = QVBoxLayout()
        right_panel.setContentsMargins(0, 0, 0, 0)
        right_panel.setSpacing(14)
        right_panel.addWidget(self.status_strip)
        right_panel.addWidget(self.telemetry_panel, stretch=1)

        right_frame = QFrame()
        right_frame.setObjectName("sidePanel")
        right_frame.setLayout(right_panel)

        content = QHBoxLayout()
        content.setContentsMargins(18, 18, 18, 18)
        content.setSpacing(18)
        content.addWidget(self.video_canvas, stretch=5)
        content.addWidget(right_frame, stretch=2)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(top_frame)
        layout.addLayout(content, stretch=1)

    def start_view(self, view: ViewDefinition) -> None:
        self._active_view = view
        self._view_name.setText(view.name.upper())
        self._playback_status.setText("LOADING")
        self._playback_status.setObjectName("statusWarning")
        self._playback_status.style().unpolish(self._playback_status)
        self._playback_status.style().polish(self._playback_status)
        self.status_strip.set_view(view)
        self.telemetry_panel.reset()
        self.video_canvas.start(video_path=view.video_path, view_name=view.name)

    def stop(self) -> None:
        self.video_canvas.stop()

    def request_back(self) -> None:
        self.back_requested.emit()

    def _on_playback_changed(
        self,
        state: str,
        current_ms: int,
        duration_ms: int,
        frame_index: int,
        fps: float,
    ) -> None:
        self._playback_status.setText(state.upper())
        self._playback_status.setObjectName("statusReady" if state == "playing" else "statusWarning")
        self._playback_status.style().unpolish(self._playback_status)
        self._playback_status.style().polish(self._playback_status)
        self._time_label.setText(f"{format_ms(current_ms)} / {format_ms(duration_ms)}")
        self.telemetry_panel.update_playback(
            playback_fps=fps,
            frame_index=frame_index,
            progress_percent=self._progress(current_ms, duration_ms),
        )

    def _on_video_finished(self) -> None:
        self._playback_status.setText("FINISHED")
        self.hud_finished.emit()

    @staticmethod
    def _progress(current_ms: int, duration_ms: int) -> float:
        if duration_ms <= 0:
            return 0.0
        return min(100.0, max(0.0, current_ms / duration_ms * 100.0))
