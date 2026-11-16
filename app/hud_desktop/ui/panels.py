from __future__ import annotations

import math

from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QProgressBar, QVBoxLayout

from app.hud_desktop.state.view_definitions import ViewDefinition
from project.runtime.pipeline_runner import FrameTelemetry


class StatusStrip(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("statusStrip")
        self._video = QLabel("NO SOURCE")
        self._backend = QLabel("WAITING")
        grid = QGridLayout()
        grid.addWidget(QLabel("VIDEO"), 0, 0)
        grid.addWidget(self._video, 0, 1)
        grid.addWidget(QLabel("ANALYSIS"), 1, 0)
        grid.addWidget(self._backend, 1, 1)
        title = QLabel("SYSTEM STATUS")
        title.setObjectName("panelTitle")
        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addLayout(grid)

    def set_view(self, view: ViewDefinition) -> None:
        self._video.setText(view.video_path.name if view.video_path.exists() else "MISSING")
        self._backend.setText("STARTING")

    def update_status(self, telemetry: FrameTelemetry) -> None:
        self._backend.setText("ACTIVE" if telemetry.tracking is not None else "PLAYBACK ONLY")


class TelemetryPanel(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("telemetryPanel")
        title = QLabel("TELEMETRY (ESTIMATES)")
        title.setObjectName("panelTitle")
        grid = QGridLayout()
        self._values = {}
        for row, (key, label) in enumerate((
            ("source_fps", "SOURCE FPS"), ("analysis_fps", "ANALYSIS FPS"),
            ("frame", "FRAME"), ("target", "TARGET"),
            ("los", "LOS [deg] x, y"), ("los_rate", "LOS RATE [deg/s] x, y"),
            ("depth", "Z EST. [m]"), ("speed", "-dZ/dt EST. [m/s]"),
            ("ttc", "TTC CONDITIONAL [s]"), ("accel", "PN EST. [m/s²] x, y"),
        )):
            name, value = QLabel(label), QLabel("--")
            name.setObjectName("fieldLabel")
            value.setObjectName("fieldValue")
            grid.addWidget(name, row, 0)
            grid.addWidget(value, row, 1)
            self._values[key] = value
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setTextVisible(False)
        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addLayout(grid)
        layout.addStretch(1)
        layout.addWidget(QLabel("PLAYBACK PROGRESS"))
        layout.addWidget(self._progress)

    def reset(self) -> None:
        for value in self._values.values():
            value.setText("--")
        self._progress.setValue(0)

    def update_telemetry(self, telemetry: FrameTelemetry) -> None:
        def number(value, *, degrees=False):
            if value is None:
                return "--"
            return f"{math.degrees(value) if degrees else value:.2f}"

        def pair(x, y, *, degrees=False):
            return f"{number(x, degrees=degrees)}, {number(y, degrees=degrees)}"

        self._values["source_fps"].setText(
            f"{telemetry.source_fps:.2f}" + (" (est.)" if telemetry.fps_is_estimated else "")
        )
        self._values["analysis_fps"].setText(number(telemetry.analysis_fps))
        self._values["frame"].setText(str(telemetry.frame_id))
        track, nav = telemetry.tracking, telemetry.navigation
        if track is None or track.state is None:
            target = "NO DATA"
        else:
            target = f"{'DETECTION' if track.detected else 'PREDICTION'} ID {track.target_id}"
        self._values["target"].setText(target)
        self._values["los"].setText(pair(nav.los_x, nav.los_y, degrees=True) if nav else "--")
        self._values["los_rate"].setText(pair(nav.los_rate_x, nav.los_rate_y, degrees=True) if nav else "--")
        self._values["depth"].setText(number(nav.depth_z) if nav else "--")
        self._values["speed"].setText(number(nav.closing_speed) if nav else "--")
        self._values["ttc"].setText(number(nav.ttc) if nav else "--")
        self._values["accel"].setText(pair(nav.accel_x, nav.accel_y) if nav else "--")
        self._progress.setValue(round(telemetry.progress_percent))
