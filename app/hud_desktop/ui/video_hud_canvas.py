from __future__ import annotations

import math

import cv2
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget


from project.runtime.pipeline_runner import FrameTelemetry


class VideoHudCanvas(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("videoHudCanvas")
        self.setMinimumSize(760, 520)
        self._frame: QImage | None = None
        self._view_name = ""
        self._progress = 0.0
        self._state = "idle"
        self._telemetry: FrameTelemetry | None = None
        self._error = ""

    def reset(self, view_name: str) -> None:
        self._frame = None
        self._view_name = view_name
        self._progress = 0.0
        self._state = "loading"
        self._telemetry = None
        self._error = ""
        self.update()

    def show_error(self, message: str = "") -> None:
        self._frame = None
        self._state = "error"
        self._telemetry = None
        self._error = message
        self.update()

    def show_frame(self, telemetry: FrameTelemetry) -> None:
        rgb = cv2.cvtColor(telemetry.frame, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb.shape
        self._frame = QImage(
            rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888
        ).copy()
        self._progress = telemetry.progress_percent / 100
        self._telemetry = telemetry
        self._state = "playing"
        self.update()

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
        self._draw_tracking(painter, target_rect)
        painter.end()
        super().paintEvent(event)

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
        text = f"ANALYSIS ERROR\n{self._error}" if self._state == "error" else "WAITING FOR VIDEO SOURCE"
        painter.drawText(rect.adjusted(20, 20, -20, -20),
                         Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, text)
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

        painter.setPen(QColor("#d6a84f"))
        painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        painter.drawText(rect.adjusted(18, 16, -18, -16), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight, self._view_name.upper())
        painter.restore()

    def _draw_tracking(self, painter: QPainter, rect: QRectF) -> None:
        telemetry = self._telemetry
        if telemetry is None or self._frame is None:
            return
        track = telemetry.tracking
        if track is None or track.state is None:
            return
        x, y, w, h = track.state[:4]
        sx, sy = rect.width() / self._frame.width(), rect.height() / self._frame.height()
        box = QRectF(rect.left() + (x - w / 2) * sx, rect.top() + (y - h / 2) * sy,
                     w * sx, h * sy)
        painter.save()
        painter.setClipRect(rect)
        color = QColor("#4fb477" if track.detected else "#d6a84f")
        pen = QPen(color, 2)
        if not track.detected:
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawRect(box)
        painter.drawText(QPointF(box.left(), max(rect.top() + 16, box.top() - 5)),
                         f"{'DETECTION' if track.detected else 'PREDICTION'} ID {track.target_id}")
        nav = telemetry.navigation
        if nav is not None and nav.accel_x is not None and nav.accel_y is not None:
            start = rect.center()
            dx, dy = nav.accel_x * 10, nav.accel_y * 10
            length = math.hypot(dx, dy)
            if length > 120:
                dx, dy = dx * 120 / length, dy * 120 / length
            end = QPointF(start.x() + dx, start.y() + dy)
            painter.setPen(QPen(QColor("#56c4eb"), 2))
            painter.drawLine(start, end)
            if length > 1:
                angle = math.atan2(dy, dx)
                for offset in (-0.5, 0.5):
                    painter.drawLine(end, QPointF(end.x() - 10 * math.cos(angle + offset),
                                                 end.y() - 10 * math.sin(angle + offset)))
            painter.drawText(QPointF(start.x() + 8, start.y() + 18), "PN (scaled)")
        painter.restore()
