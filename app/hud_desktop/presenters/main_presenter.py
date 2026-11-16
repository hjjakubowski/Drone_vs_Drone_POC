from __future__ import annotations

import threading
import time
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal, Slot

from app.hud_desktop.state.view_definitions import ViewDefinition, load_view_definitions
from app.hud_desktop.ui.main_window import MainWindow
from project.runtime.pipeline_runner import PipelineRuntime


class PipelineWorker(QThread):
    """Jeden sekwencyjny potok; GUI potwierdza odbiór każdej klatki."""

    frame_ready = Signal(object, object)
    completed = Signal(object, str)

    def __init__(self, video_path: Path) -> None:
        super().__init__()
        self.video_path = video_path
        self._ack = threading.Event()
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self.requestInterruption()
        self._cancel.set()
        self._ack.set()

    def run(self) -> None:
        error = ""
        runtime = None
        try:
            runtime = PipelineRuntime()
            if not self.isInterruptionRequested():
                runtime.start(self.video_path)
            while not self.isInterruptionRequested():
                started = time.perf_counter()
                telemetry = runtime.step()
                if telemetry is None:
                    break
                self._ack.clear()
                self.frame_ready.emit(self, telemetry)
                while not self.isInterruptionRequested() and not self._ack.wait(0.05):
                    pass
                self._cancel.wait(max(0.0, telemetry.dt - (time.perf_counter() - started)))
        except Exception as exc:
            error = str(exc) or type(exc).__name__
        finally:
            if runtime is not None:
                try:
                    runtime.stop()
                except Exception as exc:
                    error = error or str(exc) or type(exc).__name__
            self.completed.emit(self, error)


class MainPresenter(QObject):
    def __init__(self, window: MainWindow) -> None:
        super().__init__(window)
        self._window = window
        self._views = {view.id: view for view in load_view_definitions()}
        self._worker: PipelineWorker | None = None
        self._workers: list[PipelineWorker] = []
        self._pending_view = None

        self._window.views_page.set_views(list(self._views.values()))
        self._window.views_page.view_selected.connect(self._start_hud)
        self._window.views_page.video_selected.connect(self._start_video)
        self._window.hud_page.back_requested.connect(self._return_to_views)
        self._window.closing.connect(self.shutdown)

    def _start_hud(self, view_id: str) -> None:
        view = self._views.get(view_id)
        if view is None:
            self._window.show_error("Selected view not found.")
            return
        self._open_view(view)

    def _start_video(self, path: str) -> None:
        video = Path(path)
        self._open_view(ViewDefinition("custom", video.stem, video, "Selected recording"))

    def _open_view(self, view: ViewDefinition) -> None:
        self.stop()
        self._window.hud_page.start_view(view)
        self._window.show_hud()
        if any(worker.isRunning() for worker in self._workers):
            self._pending_view = view
        else:
            self._launch(view.video_path)

    def _launch(self, path: Path) -> None:
        worker = PipelineWorker(path)
        self._worker = worker
        self._workers.append(worker)
        worker.frame_ready.connect(self._show_frame)
        worker.completed.connect(self._completed)
        worker.finished.connect(self._worker_finished)
        worker.start()

    @Slot(object, object)
    def _show_frame(self, worker: PipelineWorker, telemetry) -> None:
        try:
            if worker is self._worker:
                self._window.hud_page.show_frame(telemetry)
        finally:
            worker._ack.set()

    @Slot(object, str)
    def _completed(self, worker: PipelineWorker, error: str) -> None:
        if worker is not self._worker:
            return
        self._worker = None
        if error:
            self._window.hud_page.show_error(error)
        else:
            self._window.show_views()

    @Slot()
    def _worker_finished(self) -> None:
        self._workers = [worker for worker in self._workers if worker.isRunning()]
        if self._pending_view is not None and not self._workers:
            view = self._pending_view
            self._pending_view = None
            self._launch(view.video_path)

    def stop(self) -> None:
        self._pending_view = None
        self._worker = None
        for worker in self._workers:
            worker.cancel()

    def shutdown(self) -> None:
        self.stop()
        for worker in self._workers:
            worker.wait()
        self._workers.clear()

    def _return_to_views(self) -> None:
        self.stop()
        self._window.show_views()
