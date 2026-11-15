"""Sequential video decoding; timestamps refer to the recording, not wall time."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


class VideoSource:
    """Read a constant-frame-rate recording. Unknown FPS falls back to playback at 30."""

    def __init__(self, video_path: Path | str) -> None:
        self.path = Path(video_path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Video not found: {self.path}")
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            self.close()
            raise OSError(f"Cannot open video: {self.path}")
        fps = self._capture.get(cv2.CAP_PROP_FPS)
        self.fps_is_estimated = not (math.isfinite(fps) and fps > 0)
        self.fps = 30.0 if self.fps_is_estimated else fps
        self.dt = 1.0 / self.fps
        count = self._capture.get(cv2.CAP_PROP_FRAME_COUNT)
        self.frame_count = max(0, int(count)) if math.isfinite(count) else 0
        self.duration_ms = round(self.frame_count * self.dt * 1000)
        self.frame_id = 0

    def read_frame(self) -> tuple[bool, np.ndarray | None, float, int]:
        if self._capture is None:
            return False, None, self.dt, self.frame_id
        success, frame = self._capture.read()
        if success:
            self.frame_id += 1
        else:
            self.close()
        return success, frame, self.dt, self.frame_id

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
