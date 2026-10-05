from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget


class VideoHudCanvas(QWidget):
    finished = Signal()
    playback_changed = Signal(str, int, int, int, float)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("videoHudCanvas")
        self.setMinimumSize(760, 520)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._read_next_frame)

        self._capture: cv2.VideoCapture | None = None
        self._frame: QImage | None = None
        self._video_path: Path | None = None
        self._view_name = ""
        self._fps = 30.0
        self._frame_count = 0
        self._frame_index = 0
        self._duration_ms = 0
        self._started_at = 0.0
        self._last_tick = 0.0
        self._measured_fps = 0.0
        self._state = "idle"

    def start(self, video_path: Path, view_name: str) -> None:
        self.stop()
        self._video_path = video_path
        self._view_name = view_name
        self._state = "loading"
        self._frame_index = 0
        self._frame = None

        if not video_path.exists():
            self._state = "error"
            self.playback_changed.emit("error", 0, 0, 0, 0.0)
            self.update()
            return

        self._capture = cv2.VideoCapture(str(video_path))
        if not self._capture.isOpened():
            self._state = "error"
            self.playback_changed.emit("error", 0, 0, 0, 0.0)
            self.update()
            return

        detected_fps = self._capture.get(cv2.CAP_PROP_FPS)
        self._fps = detected_fps if detected_fps and detected_fps > 1 else 30.0
        self._frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._duration_ms = int(self._frame_count / self._fps * 1000) if self._frame_count else 0
        self._started_at = time.perf_counter()
        self._last_tick = self._started_at
        self._measured_fps = 0.0
        self._state = "playing"

        interval_ms = max(1, int(1000 / self._fps))
        self._timer.start(interval_ms)
        self._read_next_frame()

    def stop(self) -> None:
        self._timer.stop()
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._state = "idle"

    def paintEvent(self, event) -> None:  # noqa: ANN001
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#050808"))

        target_rect = self._video_target_rect()
        if self._frame is None:
            self._draw_empty_state(painter, target_rect)
        else:
            painter.drawImage(target_rect, self._frame)
            self._draw_vignette(painter, target_rect)

        self._draw_hud_overlay(painter, target_rect)
        self._draw_trajectory_placeholder(painter, target_rect)
        painter.end()
        super().paintEvent(event)

    def _read_next_frame(self) -> None:
        if self._capture is None:
            return

        success, frame = self._capture.read()
        if not success:
            self._timer.stop()
            self._state = "finished"
            self.playback_changed.emit(
                "finished",
                self._duration_ms,
                self._duration_ms,
                self._frame_index,
                self._measured_fps,
            )
            self.finished.emit()
            return

        now = time.perf_counter()
        delta = now - self._last_tick
        self._last_tick = now
        if delta > 0:
            instant_fps = 1.0 / delta
            self._measured_fps = instant_fps if self._measured_fps == 0 else self._measured_fps * 0.85 + instant_fps * 0.15

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = frame_rgb.shape
        bytes_per_line = channels * width
        self._frame = QImage(
            frame_rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()

        self._frame_index += 1
        current_ms = int(self._frame_index / self._fps * 1000)
        self.playback_changed.emit(
            "playing",
            current_ms,
            self._duration_ms,
            self._frame_index,
            self._measured_fps or self._fps,
        )
        self.update()

    def _video_target_rect(self) -> QRectF:
        bounds = QRectF(self.rect()).adjusted(14, 14, -14, -14)
        if self._frame is None or self._frame.width() <= 0 or self._frame.height() <= 0:
            return bounds

        image_ratio = self._frame.width() / self._frame.height()
        bounds_ratio = bounds.width() / bounds.height()

        if image_ratio > bounds_ratio:
            width = bounds.width()
            height = width / image_ratio
        else:
            height = bounds.height()
            width = height * image_ratio

        x = bounds.x() + (bounds.width() - width) / 2
        y = bounds.y() + (bounds.height() - height) / 2
        return QRectF(x, y, width, height)

    def _draw_empty_state(self, painter: QPainter, rect: QRectF) -> None:
        painter.save()
        painter.setPen(QPen(QColor("#4d6262"), 1))
        painter.drawRect(rect)
        painter.setPen(QColor("#9fb2b2"))
        painter.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        text = "VIDEO SOURCE NOT AVAILABLE" if self._state == "error" else "WAITING FOR VIDEO SOURCE"
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
        painter.restore()

    def _draw_vignette(self, painter: QPainter, rect: QRectF) -> None:
        painter.save()
        painter.setPen(QPen(QColor(0, 0, 0, 90), 16))
        painter.drawRect(rect.adjusted(8, 8, -8, -8))
        painter.restore()

    def _draw_hud_overlay(self, painter: QPainter, rect: QRectF) -> None:
        painter.save()
        green = QColor("#4fb477")
        muted = QColor("#6f8383")
        warning = QColor("#d6a84f")

        painter.setPen(QPen(green, 1))
        painter.drawRect(rect)

        center = rect.center()
        painter.drawLine(QPointF(center.x() - 22, center.y()), QPointF(center.x() + 22, center.y()))
        painter.drawLine(QPointF(center.x(), center.y() - 22), QPointF(center.x(), center.y() + 22))

        corner = 42
        for start, end in (
            (rect.topLeft(), QPointF(rect.left() + corner, rect.top())),
            (rect.topLeft(), QPointF(rect.left(), rect.top() + corner)),
            (rect.topRight(), QPointF(rect.right() - corner, rect.top())),
            (rect.topRight(), QPointF(rect.right(), rect.top() + corner)),
            (rect.bottomLeft(), QPointF(rect.left() + corner, rect.bottom())),
            (rect.bottomLeft(), QPointF(rect.left(), rect.bottom() - corner)),
            (rect.bottomRight(), QPointF(rect.right() - corner, rect.bottom())),
            (rect.bottomRight(), QPointF(rect.right(), rect.bottom() - corner)),
        ):
            painter.drawLine(start, end)

        painter.setPen(QPen(muted, 1, Qt.PenStyle.DotLine))
        third_w = rect.width() / 3
        third_h = rect.height() / 3
        painter.drawLine(QPointF(rect.left() + third_w, rect.top()), QPointF(rect.left() + third_w, rect.bottom()))
        painter.drawLine(QPointF(rect.left() + third_w * 2, rect.top()), QPointF(rect.left() + third_w * 2, rect.bottom()))
        painter.drawLine(QPointF(rect.left(), rect.top() + third_h), QPointF(rect.right(), rect.top() + third_h))
        painter.drawLine(QPointF(rect.left(), rect.top() + third_h * 2), QPointF(rect.right(), rect.top() + third_h * 2))

        painter.setPen(warning)
        painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        painter.drawText(rect.adjusted(18, 16, -18, -16), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, "TRAJECTORY PLACEHOLDER")
        painter.drawText(rect.adjusted(18, 16, -18, -16), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight, self._view_name.upper())
        painter.restore()

    def _draw_trajectory_placeholder(self, painter: QPainter, rect: QRectF) -> None:
        painter.save()
        path_pen = QPen(QColor("#d6a84f"), 2)
        path_pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(path_pen)

        points = self._trajectory_points(rect)
        for index in range(len(points) - 1):
            painter.drawLine(points[index], points[index + 1])

        painter.setPen(QPen(QColor("#d6a84f"), 1))
        painter.setBrush(QColor(214, 168, 79, 60))
        for point in points:
            painter.drawEllipse(point, 4, 4)

        if points:
            painter.setPen(QPen(QColor("#d95c5c"), 2))
            painter.setBrush(QColor(217, 92, 92, 70))
            painter.drawEllipse(points[-1], 12, 12)

        painter.restore()

    def _trajectory_points(self, rect: QRectF) -> list[QPointF]:
        progress = 0.0
        if self._duration_ms > 0:
            progress = min(1.0, max(0.0, (self._frame_index / self._fps * 1000) / self._duration_ms))

        points: list[QPointF] = []
        for step in range(8):
            t = step / 7
            x = rect.left() + rect.width() * (0.16 + 0.68 * t)
            wave = math.sin((t + progress * 0.35) * math.pi * 1.6)
            y = rect.top() + rect.height() * (0.68 - 0.34 * t + 0.07 * wave)
            points.append(QPointF(x, y))
        return points
