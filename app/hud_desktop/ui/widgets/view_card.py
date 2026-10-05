from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout

from app.hud_desktop.state.view_definitions import ViewDefinition


class ViewCard(QFrame):
    clicked = Signal(str)

    def __init__(self, view: ViewDefinition) -> None:
        super().__init__()
        self._view = view
        self.setObjectName("viewCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(210)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        name = QLabel(view.name)
        name.setObjectName("viewCardTitle")

        status = QLabel(view.status)
        status.setObjectName("viewCardStatusReady" if view.status == "READY" else "viewCardStatusError")

        video = QLabel(view.video_path.name)
        video.setObjectName("viewCardVideo")

        description = QLabel(view.description)
        description.setObjectName("viewCardDescription")
        description.setWordWrap(True)

        meta = QLabel("AUTO START  |  HUD OVERLAY  |  PLACEHOLDER TRAJECTORY")
        meta.setObjectName("viewCardMeta")

        top = QHBoxLayout()
        top.addWidget(name)
        top.addStretch(1)
        top.addWidget(status)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 20)
        layout.setSpacing(12)
        layout.addLayout(top)
        layout.addWidget(video)
        layout.addWidget(description)
        layout.addStretch(1)
        layout.addWidget(meta)

    def mouseReleaseEvent(self, event) -> None:  # noqa: ANN001
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._view.id)
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: ANN001
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.clicked.emit(self._view.id)
            return
        super().keyPressEvent(event)
