from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from app.hud_desktop.state.view_definitions import ViewDefinition
from app.hud_desktop.ui.panels import StatusStrip, TelemetryPanel
from app.hud_desktop.ui.video_hud_canvas import VideoHudCanvas
from project.runtime.pipeline_runner import FrameTelemetry


class HudPage(QWidget):
    back_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("hudPage")

        self._view_name = QLabel("VIEW")
        self._view_name.setObjectName("hudTitle")
        self._playback_status = QLabel("IDLE")
        self._playback_status.setObjectName("statusReady")
        self._time_label = QLabel("00:00 / 00:00")
        self._time_label.setObjectName("hudTime")

        self._back_button = QPushButton("Back")
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
        self._view_name.setText(view.name.upper())
        self._set_status("LOADING", "statusWarning")
        self._time_label.setText("00:00 / 00:00")
        self.status_strip.set_view(view)
        self.telemetry_panel.reset()
        self.video_canvas.reset(view.name)

    def show_frame(self, telemetry: FrameTelemetry) -> None:
        self.video_canvas.show_frame(telemetry)
        self._set_status("PLAYING", "statusReady")
        self._time_label.setText(
            f"{format_ms(telemetry.current_ms)} / {format_ms(telemetry.duration_ms)}"
        )
        self.status_strip.update_status(telemetry)
        self.telemetry_panel.update_telemetry(telemetry)

    def show_error(self, message: str) -> None:
        self._set_status("ERROR", "statusError")
        self._playback_status.setToolTip(message)
        self.video_canvas.show_error(message)

    def request_back(self) -> None:
        self.back_requested.emit()

    def _set_status(self, text: str, style: str) -> None:
        self._playback_status.setText(text)
        self._playback_status.setToolTip("")
        if self._playback_status.objectName() != style:
            self._playback_status.setObjectName(style)
            self._playback_status.style().unpolish(self._playback_status)
            self._playback_status.style().polish(self._playback_status)


def format_ms(milliseconds: int | float) -> str:
    total_seconds = max(0, int(milliseconds // 1000))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"
