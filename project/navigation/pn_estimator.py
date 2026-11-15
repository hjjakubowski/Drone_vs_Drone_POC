"""Simplified navigation estimates for video analysis, without flight control.

Angles are relative to the moving camera. Depth, speed and TTC are conditional
approximations that depend on the target's orientation and size, among others.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from numbers import Integral, Real
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from project.vision.tracker_engine import TrackingResult


@dataclass(frozen=True)
class NavigationResult:
    """Optional values: angles [rad], angular rates [rad/s], Z [m],
    depth rate [m/s], TTC [s] and acceleration indications [m/s²].

    Image axes: x points right, y points down. `closing_speed` approximates -dZ/dt.
    """

    target_id: int | None
    fps_is_estimated: bool
    los_x: float | None = None
    los_y: float | None = None
    los_rate_x: float | None = None
    los_rate_y: float | None = None
    depth_z: float | None = None
    closing_speed: float | None = None
    ttc: float | None = None
    accel_x: float | None = None
    accel_y: float | None = None


def _number(config: dict, key: str, *, positive: bool = False,
            nonnegative: bool = False) -> float:
    value = config.get(key)
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value)
            or (positive and value <= 0) or (nonnegative and value < 0)):
        raise ValueError(f"Invalid navigation parameter: {key}={value!r}")
    return float(value)


def _positive_int(config: dict, key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"Parameter {key} must be a positive integer")
    return int(value)


class ProportionalNavigation:
    """Image-based estimator with a short history for one target's consecutive frames."""

    def __init__(self, config: dict) -> None:
        self.image_size = (_positive_int(config, "image_width"),
                           _positive_int(config, "image_height"))
        self.fx = _number(config, "fx", positive=True)
        self.fy = _number(config, "fy", positive=True)
        self.cx = _number(config, "cx")
        self.cy = _number(config, "cy")
        self.target_width_m = _number(config, "target_width_m", positive=True)
        self.guidance_gain = _number(config, "guidance_gain", nonnegative=True)
        self.ttc_denominator_min = _number(config, "ttc_denominator_min", nonnegative=True)
        self._speeds: deque[float] = deque(maxlen=_positive_int(config, "closing_speed_window"))
        self.reset()

    def reset(self) -> None:
        """Clear the previous target, measurements and speed window."""
        self._previous: tuple[int, int, float, float, float] | None = None
        self._speeds.clear()

    def process(self, tracking: TrackingResult, dt: float, frame_id: int,
                image_size: tuple[int, int], fps_is_estimated: bool) -> NavigationResult:
        """Process one consecutive frame; image_size is (width, height)."""
        if (isinstance(dt, bool) or not isinstance(dt, Real)
                or not math.isfinite(dt) or dt <= 0):
            self.reset()
            raise ValueError("dt must be a positive finite recording-time interval")
        if (not isinstance(image_size, (tuple, list)) or len(image_size) != 2
                or any(isinstance(n, bool) or not isinstance(n, Integral) or n <= 0
                       for n in image_size) or tuple(image_size) != self.image_size):
            self.reset()
            raise ValueError(f"Expected frame size {self.image_size}, got {image_size!r}")

        empty = NavigationResult(target_id=tracking.target_id, fps_is_estimated=fps_is_estimated)
        if not tracking.detected or tracking.target_id is None or tracking.state is None:
            self.reset()
            return empty
        try:
            state = np.asarray(tracking.state, dtype=float)
        except (TypeError, ValueError):
            self.reset()
            return empty
        if state.shape != (8,) or not np.isfinite(state).all() or state[2] <= 0 or state[3] <= 0:
            self.reset()
            return empty

        xc, yc, w, h, _, _, vw, vh = state
        los_x = math.atan((xc - self.cx) / self.fx)
        los_y = math.atan((yc - self.cy) / self.fy)
        depth_z = self.fx * self.target_width_m / w
        previous = self._previous
        if previous is not None and (previous[0] != tracking.target_id or previous[1] + 1 != frame_id):
            self.reset()
            previous = None
        self._previous = (tracking.target_id, frame_id, los_x, los_y, depth_z)
        if previous is None:
            return NavigationResult(tracking.target_id, fps_is_estimated,
                                    los_x=los_x, los_y=los_y, depth_z=depth_z)

        los_rate_x = (los_x - previous[2]) / dt
        los_rate_y = (los_y - previous[3]) / dt
        self._speeds.append((previous[4] - depth_z) / dt)
        closing_speed = sum(self._speeds) / len(self._speeds)
        denom = w * (vh / dt) + h * (vw / dt)
        ttc = 2 * w * h / denom if denom > self.ttc_denominator_min else None
        accel_x = self.guidance_gain * closing_speed * los_rate_x if closing_speed > 0 else None
        accel_y = self.guidance_gain * closing_speed * los_rate_y if closing_speed > 0 else None
        return NavigationResult(
            tracking.target_id, fps_is_estimated, los_x, los_y, los_rate_x, los_rate_y,
            depth_z, closing_speed, ttc, accel_x, accel_y,
        )
