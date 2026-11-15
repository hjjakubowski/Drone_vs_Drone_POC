from __future__ import annotations

from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QProgressBar, QVBoxLayout

from app.hud_desktop.state.view_definitions import ViewDefinition


class StatusStrip(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("statusStrip")

        self._video = self._value("NO SOURCE")
        self._backend = self._value("PLACEHOLDER")
        self._trajectory = self._value("WAITING")
        self._link = self._value("SIMULATED")

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(10)
        grid.addWidget(self._label("VIDEO"), 0, 0)
        grid.addWidget(self._video, 0, 1)
        grid.addWidget(self._label("BACKEND"), 1, 0)
        grid.addWidget(self._backend, 1, 1)
        grid.addWidget(self._label("TRAJECTORY"), 2, 0)
        grid.addWidget(self._trajectory, 2, 1)
        grid.addWidget(self._label("LINK"), 3, 0)
        grid.addWidget(self._link, 3, 1)

        title = QLabel("SYSTEM STATUS")
        title.setObjectName("panelTitle")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)
        layout.addWidget(title)
        layout.addLayout(grid)

    def set_view(self, view: ViewDefinition) -> None:
        self._video.setText(view.video_path.name if view.video_path.exists() else "MISSING")
        self._backend.setText("FRONTEND MOCK")
        self._trajectory.setText("PLACEHOLDER")
        self._link.setText("SIMULATED")

    @staticmethod
    def _label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("fieldLabel")
        return label

    @staticmethod
    def _value(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("fieldValue")
        return label


class TelemetryPanel(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("telemetryPanel")

        self._fps = self._value("--")
        self._frame = self._value("--")
        self._target = self._value("NO TARGET")
        self._trajectory = self._value("NOT COMPUTED")
        self._command = self._value("N/A")
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(12)
        grid.addWidget(self._label("FPS"), 0, 0)
        grid.addWidget(self._fps, 0, 1)
        grid.addWidget(self._label("FRAME"), 1, 0)
        grid.addWidget(self._frame, 1, 1)
        grid.addWidget(self._label("TARGET"), 2, 0)
        grid.addWidget(self._target, 2, 1)
        grid.addWidget(self._label("TRAJECTORY"), 3, 0)
        grid.addWidget(self._trajectory, 3, 1)
        grid.addWidget(self._label("COMMAND"), 4, 0)
        grid.addWidget(self._command, 4, 1)

        title = QLabel("TELEMETRY")
        title.setObjectName("panelTitle")

        progress_label = QLabel("PLAYBACK PROGRESS")
        progress_label.setObjectName("fieldLabel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)
        layout.addWidget(title)
        layout.addLayout(grid)
        layout.addStretch(1)
        layout.addWidget(progress_label)
        layout.addWidget(self._progress)

    def reset(self) -> None:
        self._fps.setText("--")
        self._frame.setText("--")
        self._target.setText("NO TARGET")
        self._trajectory.setText("PLACEHOLDER")
        self._command.setText("WAITING")
        self._progress.setValue(0)

    def update_playback(self, playback_fps: float, frame_index: int, progress_percent: float) -> None:
        self._fps.setText(f"{playback_fps:.1f}")
        self._frame.setText(str(frame_index))
        self._target.setText("PENDING BACKEND")
        self._trajectory.setText("PLACEHOLDER")
        self._command.setText("N/A")
        self._progress.setValue(round(progress_percent))

    @staticmethod
    def _label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("fieldLabel")
        return label

    @staticmethod
    def _value(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("fieldValue")
        return label
