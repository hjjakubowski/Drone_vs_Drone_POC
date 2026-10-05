from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from app.hud_desktop.presenters.main_presenter import MainPresenter
from app.hud_desktop.ui.main_window import MainWindow
from app.hud_desktop.utils.paths import resource_path


def _load_stylesheet() -> str:
    stylesheet_path = resource_path("app/hud_desktop/resources/styles/main.qss")
    if not stylesheet_path.exists():
        return ""
    return stylesheet_path.read_text(encoding="utf-8")


def run(argv: list[str] | None = None) -> int:
    app = QApplication(argv or sys.argv)
    app.setApplicationName("INZ HUD")
    app.setOrganizationName("INZ")
    app.setStyleSheet(_load_stylesheet())

    window = MainWindow()
    presenter = MainPresenter(window)
    window.setProperty("_presenter", presenter)
    window.resize(1360, 820)
    window.show()

    return app.exec()
