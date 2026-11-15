"""Single-target detection and tracking with YOLO and BoT-SORT."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from project.config import load_config, project_root
from project.vision.engine_metadata import read_engine_metadata


@dataclass(frozen=True)
class TrackingResult:
    """Result for one frame.

    state contains:
        [xc, yc, w, h, vx, vy, vw, vh]

    Coordinates and dimensions are in pixels.
    Velocity components are in pixels per tracker step.
    """

    target_id: int | None
    state: np.ndarray | None
    detected: bool
    missing_frames: int


class VisionTracker:
    def __init__(
        self,
        weights_path: Path | str,
        *,
        max_missing_frames: int | None = None,
        device: str = "cuda:0",
    ) -> None:
        weights_path = Path(weights_path)
        if not weights_path.is_file():
            raise FileNotFoundError(f"Model not found: {weights_path}")

        config = load_config("model")["model"]
        if max_missing_frames is None:
            max_missing_frames = config["max_missing_frames"]
        if max_missing_frames < 0:
            raise ValueError("max_missing_frames must not be negative")
        tracker_path = project_root() / config["tracker"]

        if not tracker_path.is_file():
            raise FileNotFoundError(
                f"Tracker configuration not found: {tracker_path}"
            )

        self.model = YOLO(str(weights_path), task="detect")
        self.tracker_path = str(tracker_path)
        self.imgsz = (
            read_engine_metadata(weights_path)["imgsz"]
            if weights_path.suffix == ".engine" else config["imgsz"]
        )
        self.rect = config["rect"]
        self.quantize = config["quantize"]
        self.classes = config["classes"]
        self.confidence_threshold = config["confidence_threshold"]
        self.device = device
        self.max_missing_frames = max_missing_frames

        self.active_target_id: int | None = None
        self.missing_frames = 0

    def process_frame(self, frame: np.ndarray) -> TrackingResult:
        if frame is None or frame.size == 0:
            raise ValueError("Frame is empty")

        result = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_path,
            classes=self.classes,
            imgsz=self.imgsz,
            rect=self.rect,
            quantize=self.quantize,
            device=self.device,
            # Allow low-confidence detections into the second BoT-SORT stage.
            conf=self.confidence_threshold,
            verbose=False,
        )[0]

        boxes = result.boxes
        if boxes is None or boxes.id is None:
            visible_ids: list[int] = []
        else:
            visible_ids = boxes.id.int().cpu().tolist()

        if self.active_target_id is None and visible_ids:
            self.active_target_id = visible_ids[0]

        if self.active_target_id in visible_ids:
            self.missing_frames = 0

            # Access the internal BoT-SORT object.
            tracker = self.model.predictor.trackers[0]
            track = next(
                (
                    track
                    for track in tracker.tracked_stracks
                    if track.track_id == self.active_target_id
                    and track.is_activated
                ),
                None,
            )

            if track is None or track.mean is None:
                raise RuntimeError(
                    "Visible ID has no corresponding BoT-SORT state."
                )

            state = np.asarray(track.mean, dtype=float).copy()
            if state.shape != (8,):
                raise RuntimeError(
                    f"Unexpected BoT-SORT state shape: {state.shape}"
                )

            return TrackingResult(
                target_id=self.active_target_id,
                state=state,
                detected=True,
                missing_frames=0,
            )

        if self.active_target_id is not None:
            self.missing_frames += 1

            if self.missing_frames > self.max_missing_frames:
                self.active_target_id = None
                self.missing_frames = 0
            else:
                tracker = self.model.predictor.trackers[0]
                track = next(
                    (t for t in tracker.lost_stracks
                     if t.track_id == self.active_target_id and t.mean is not None),
                    None,
                )
                if track is not None:
                    state = np.asarray(track.mean, dtype=float).copy()
                    if state.shape != (8,):
                        raise RuntimeError(f"Unexpected BoT-SORT state shape: {state.shape}")
                    return TrackingResult(
                        target_id=self.active_target_id,
                        state=state,
                        detected=False,
                        missing_frames=self.missing_frames,
                    )

        return TrackingResult(
            target_id=self.active_target_id,
            state=None,
            detected=False,
            missing_frames=self.missing_frames,
        )

    def reset(self) -> None:
        """Call before analyzing a different recording."""
        predictor = self.model.predictor
        if predictor is not None:
            for tracker in getattr(predictor, "trackers", []):
                tracker.reset()

        self.active_target_id = None
        self.missing_frames = 0
