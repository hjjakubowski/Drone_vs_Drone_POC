"""Sequential playback pipeline. Detection and PN are not connected yet."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from project.video.video_source import VideoSource


@dataclass(frozen=True)
class FrameTelemetry:
    """GUI-independent frame and playback measurements; no fabricated AI results."""

    frame: np.ndarray
    frame_id: int
    dt: float
    current_ms: int
    duration_ms: int
    source_fps: float
    playback_fps: float

    @property
    def progress_percent(self) -> float:
        if self.duration_ms <= 0:
            return 0.0
        return min(100.0, max(0.0, self.current_ms / self.duration_ms * 100))


class PipelineRuntime:
    """Own the decoder and advance one frame per call, without depending on Qt."""

    def __init__(self) -> None:
        self.source: VideoSource | None = None
        self._last_tick: float | None = None
        self._playback_fps = 0.0

    def start(self, video_path: Path | str) -> None:
        self.stop()
        self.source = VideoSource(video_path)

    def step(self) -> FrameTelemetry | None:
        if self.source is None:
            return None
        success, frame, dt, frame_id = self.source.read_frame()
        if not success:
            self.stop()
            return None
        now = time.perf_counter()
        if self._last_tick is not None and now > self._last_tick:
            fps = 1.0 / (now - self._last_tick)
            self._playback_fps = self._playback_fps * 0.85 + fps * 0.15
        else:
            self._playback_fps = self.source.fps
        self._last_tick = now
        return FrameTelemetry(
            frame=frame,
            frame_id=frame_id,
            dt=dt,
            current_ms=round(frame_id * dt * 1000),
            duration_ms=self.source.duration_ms,
            source_fps=self.source.fps,
            playback_fps=self._playback_fps,
        )

    def stop(self) -> None:
        if self.source is not None:
            self.source.close()
            self.source = None
        self._last_tick = None
        self._playback_fps = 0.0
