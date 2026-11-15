"""Compare tracker coordinates for the same frame at two resolutions.

Requires TensorRT, CUDA, and a 1920×1080 video containing a drone.
Does not require a GUI or write output files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from project.config import load_config, project_root
from project.video.video_source import VideoSource
from project.vision.tracker_engine import VisionTracker


def xywh_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two boxes [xc, yc, w, h] in the same coordinate system."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != (4,) or b.shape != (4,) or not np.isfinite([*a, *b]).all():
        raise ValueError("Boxes must contain four finite values")
    if min(a[2], a[3], b[2], b[3]) <= 0:
        raise ValueError("Box dimensions must be positive")
    left_top = np.maximum(a[:2] - a[2:] / 2, b[:2] - b[2:] / 2)
    right_bottom = np.minimum(a[:2] + a[2:] / 2, b[:2] + b[2:] / 2)
    overlap = np.maximum(right_bottom - left_top, 0)
    intersection = float(overlap[0] * overlap[1])
    return intersection / float(a[2] * a[3] + b[2] * b[3] - intersection)


def detected_box(tracker: VisionTracker, target_id: int) -> np.ndarray:
    """Read the Results box for the ID selected by VisionTracker."""
    boxes = tracker.model.predictor.results[0].boxes
    if boxes is None or boxes.id is None:
        raise RuntimeError("Tracker returned a state, but Results contains no ID")
    ids = boxes.id.int().cpu().tolist()
    if target_id not in ids:
        raise RuntimeError(f"ID {target_id} is missing from Results")
    return boxes.xywh[ids.index(target_id)].cpu().numpy().astype(float)


def check_state_matches_box(state: np.ndarray, box: np.ndarray) -> None:
    if state.shape != (8,) or not np.isfinite(state).all():
        raise AssertionError(f"Invalid track.mean vector: {state}")
    if not np.allclose(state[:4], box, rtol=1e-5, atol=0.02):
        raise AssertionError(f"track.mean[:4] differs from Results.boxes.xywh: {state[:4]} vs {box}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="1920×1080 video containing a drone")
    parser.add_argument("--max-frames", type=int, default=300, help="Maximum number of consecutive frames (default: 300)")
    parser.add_argument("--stride", type=int, default=10, help="Check every Nth frame (default: 10)")
    parser.add_argument("--min-matches", type=int, default=3, help="Required number of detection pairs (default: 3)")
    parser.add_argument("--min-iou", type=float, default=0.6, help="Minimum IoU at the original scale (default: 0.6)")
    args = parser.parse_args()
    if args.max_frames < 1 or args.stride < 1 or args.min_matches < 1 or not 0 < args.min_iou <= 1:
        parser.error("Limits must be positive and --min-iou must be in (0, 1]")
    if not torch.cuda.is_available():
        raise RuntimeError("The TensorRT engine test requires an available NVIDIA GPU and CUDA-enabled PyTorch.")

    root = project_root()
    config = load_config("model")["model"]
    engine = root / config["engine"]
    if engine.suffix.lower() != ".engine":
        raise ValueError(f"Configured model is not an .engine file: {engine}")
    tracker = VisionTracker(engine)
    if tuple(tracker.imgsz) != (544, 960):
        raise ValueError(f"Expected engine size [544, 960], got: {tracker.imgsz}")
    source = VideoSource(args.video)
    checked = matches = skipped = 0
    failures: list[str] = []
    print(f"Engine: {engine}\nVideo: {args.video}")
    print("Comparison: 1920×1080 and the same image resized to 960×540; "
          "each run starts with an empty tracker.")
    try:
        while source.frame_id < args.max_frames:
            ok, frame, _, frame_id = source.read_frame()
            if not ok:
                break
            if frame.shape[:2] != (1080, 1920):
                raise ValueError(f"Expected a 1920×1080 video, got {frame.shape[1]}×{frame.shape[0]}")
            if (frame_id - 1) % args.stride:
                continue
            checked += 1
            tracker.reset()
            full = tracker.process_frame(frame)
            if not full.detected or full.state is None:
                skipped += 1
                print(f"Frame {frame_id}: no detection in original — skipped")
                continue
            full_box = detected_box(tracker, full.target_id)
            full_state = full.state.copy()
            tracker.reset()
            small = tracker.process_frame(cv2.resize(frame, (960, 540), interpolation=cv2.INTER_LINEAR))
            if not small.detected or small.state is None:
                skipped += 1
                print(f"Frame {frame_id}: no detection in resized image — skipped")
                continue
            small_box = detected_box(tracker, small.target_id)
            small_state = small.state.copy()
            try:
                check_state_matches_box(full_state, full_box)
                check_state_matches_box(small_state, small_box)
                iou = xywh_iou(full_box, small_box * 2)
            except (AssertionError, ValueError) as exc:
                failures.append(f"Frame {frame_id}: {exc}")
                print(failures[-1])
                continue
            matches += 1
            print(f"Frame {frame_id}: IoU={iou:.3f}; "
                  f"1920 box={np.round(full_box, 1)}; "
                  f"960 box × 2={np.round(small_box * 2, 1)}; "
                  f"mean[:4]={np.round(full_state[:4], 1)}")
            if iou < args.min_iou:
                failures.append(f"Frame {frame_id}: IoU {iou:.3f} < {args.min_iou:.3f}")
    finally:
        source.close()

    print(f"Checked: {checked}, detection pairs: {matches}, skipped: {skipped}, failures: {len(failures)}")
    if failures:
        raise SystemExit("FAILURE: comparison did not pass. "
                         "Check whether both boxes refer to the same target and review the IoU threshold.")
    if matches < args.min_matches:
        raise SystemExit(f"INCONCLUSIVE: {args.min_matches} detection pairs required, got {matches}.")
    print("SUCCESS: boxes and track.mean match the input-frame scale in the checked cases.")
    print("This test does not validate classification or localization against ground truth.")


if __name__ == "__main__":
    main()
