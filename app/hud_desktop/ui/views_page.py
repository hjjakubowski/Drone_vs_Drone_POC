from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from app.hud_desktop.state.view_definitions import ViewDefinition


class ViewsPage(QWidget):
    view_selected = Signal(str)
    video_selected = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("viewsPage")

        title = QLabel("AUTONOMOUS DRONE HUD")
        title.setObjectName("appTitle")

        subtitle = QLabel("Mission view selection")
        subtitle.setObjectName("appSubtitle")

        header_meta = QLabel("VIDEO ANALYSIS  |  YOLO + BOT-SORT  |  PN ESTIMATES")
        header_meta.setObjectName("headerMeta")

        header = QHBoxLayout()
        title_group = QVBoxLayout()
        title_group.setSpacing(4)
        title_group.addWidget(title)
        title_group.addWidget(subtitle)
        header.addLayout(title_group, stretch=1)
        header.addWidget(header_meta)

        self._cards_grid = QGridLayout()
        self._cards_grid.setHorizontalSpacing(18)
        self._cards_grid.setVerticalSpacing(18)

        cards_frame = QFrame()
        cards_frame.setObjectName("contentFrame")
        cards_frame.setLayout(self._cards_grid)

        footer = QLabel("Select a view to open the HUD.")
        footer.setObjectName("hintText")
        open_button = QPushButton("Open video…")
        open_button.clicked.connect(self._choose_video)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(42, 34, 42, 30)
        layout.setSpacing(24)
        layout.addLayout(header)
        layout.addWidget(cards_frame, stretch=1)
        layout.addWidget(open_button)
        layout.addWidget(footer)

    def _choose_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select recording", "", "Video (*.mp4 *.MP4)")
        if path:
            self.video_selected.emit(path)

    def set_views(self, views: list[ViewDefinition]) -> None:
        while self._cards_grid.count():
            item = self._cards_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for index, view in enumerate(views):
            card = ViewCard(view)
            card.clicked.connect(self.view_selected)
            row, col = divmod(index, 2)
            self._cards_grid.addWidget(card, row, col)


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

        meta = QLabel("AUTO START  |  TRACKING  |  NAVIGATION ESTIMATES")
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
