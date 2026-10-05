from __future__ import annotations

from PySide6.QtCore import QObject

from app.hud_desktop.state.view_definitions import ViewDefinition, load_view_definitions
from app.hud_desktop.ui.main_window import MainWindow


class MainPresenter(QObject):
    def __init__(self, window: MainWindow) -> None:
        super().__init__(window)
        self._window = window
        self._views = {view.id: view for view in load_view_definitions()}

        self._window.views_page.set_views(list(self._views.values()))
        self._window.views_page.view_selected.connect(self._start_hud)
        self._window.hud_page.hud_finished.connect(self._return_to_views)
        self._window.hud_page.back_requested.connect(self._return_to_views)

    def _start_hud(self, view_id: str) -> None:
        selected_view = self._views.get(view_id)
        if selected_view is None:
            self._window.show_error("Nie znaleziono wybranego widoku.")
            return

        self._window.show_hud()
        self._window.hud_page.start_view(selected_view)

    def _return_to_views(self) -> None:
        self._window.hud_page.stop()
        self._window.show_views()
