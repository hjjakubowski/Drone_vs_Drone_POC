from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QMessageBox, QStackedWidget, QVBoxLayout, QWidget

from app.hud_desktop.ui.hud_page import HudPage
from app.hud_desktop.ui.views_page import ViewsPage


class MainWindow(QWidget):
    closing = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("INZ HUD")
        self.setObjectName("mainWindow")
        self.setMinimumSize(1180, 720)

        self.views_page = ViewsPage()
        self.hud_page = HudPage()

        self._stack = QStackedWidget()
        self._stack.addWidget(self.views_page)
        self._stack.addWidget(self.hud_page)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._stack)

        self.show_views()

    def show_views(self) -> None:
        self._stack.setCurrentWidget(self.views_page)

    def show_hud(self) -> None:
        self._stack.setCurrentWidget(self.hud_page)

    def show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Blad aplikacji", message)

    def keyPressEvent(self, event) -> None:  # noqa: ANN001
        if event.key() == Qt.Key.Key_Escape and self._stack.currentWidget() is self.hud_page:
            self.hud_page.request_back()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # noqa: ANN001
        self.closing.emit()
        super().closeEvent(event)
