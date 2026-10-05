from __future__ import annotations

from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QVBoxLayout

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
