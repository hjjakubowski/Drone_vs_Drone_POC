from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from app.hud_desktop.state.view_definitions import ViewDefinition
from app.hud_desktop.ui.widgets.view_card import ViewCard


class ViewsPage(QWidget):
    view_selected = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("viewsPage")

        title = QLabel("AUTONOMOUS DRONE HUD")
        title.setObjectName("appTitle")

        subtitle = QLabel("Panel wyboru widoku misji")
        subtitle.setObjectName("appSubtitle")

        header_meta = QLabel("FRONTEND MODE  |  BACKEND PLACEHOLDER  |  VIDEO PLAYBACK")
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

        footer = QLabel("Wybierz widok, aby uruchomic ekran HUD.")
        footer.setObjectName("hintText")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(42, 34, 42, 30)
        layout.setSpacing(24)
        layout.addLayout(header)
        layout.addWidget(cards_frame, stretch=1)
        layout.addWidget(footer)

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
