"""Video analysis: VideoSource → VisionTracker → ProportionalNavigation."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2

from project.config import load_config, project_root
from project.navigation.pn_estimator import NavigationResult, ProportionalNavigation
from project.video.video_source import VideoSource
from project.vision.tracker_engine import VisionTracker


def format_value(value: float | None, *, degrees: bool = False) -> str:
    if value is None:
        return "unavailable"
    return f"{math.degrees(value) if degrees else value:.2f}"


def format_result(frame_id: int, dt: float, tracking, result: NavigationResult) -> str:
    if tracking.detected and tracking.state is not None:
        status = "detection"
    elif tracking.state is not None:
        status = f"prediction ({tracking.missing_frames} frames)"
    else:
        status = "no state"
    f = format_value
    return (
        f"Frame {frame_id:5d} | time ~{(frame_id - 1) * dt:.3f} s | "
        f"{status} | ID={tracking.target_id}\n"
        f"  LOS [deg]: x={f(result.los_x, degrees=True)}, y={f(result.los_y, degrees=True)} | "
        f"LOS rate [deg/s]: x={f(result.los_rate_x, degrees=True)}, "
        f"y={f(result.los_rate_y, degrees=True)}\n"
        f"  Z est. [m]={f(result.depth_z)} | -dZ/dt est. [m/s]={f(result.closing_speed)} | "
        f"Conditional TTC [s]={f(result.ttc)}\n"
        f"  Simplified PN [m/s²]: x={f(result.accel_x)}, y={f(result.accel_y)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="1920×1080 video containing a drone")
    parser.add_argument("--max-frames", type=int, default=0, help="Frame limit; 0 means the entire video")
    parser.add_argument("--every", type=int, default=1, help="Print every N frames; all frames are processed")
    parser.add_argument("--show", action="store_true", help="Show a video preview and the target box")
    args = parser.parse_args()
    if args.max_frames < 0 or args.every < 1:
        parser.error("--max-frames must not be negative and --every must be positive")

    root = project_root()
    model_config = load_config("model")["model"]
    nav_config = load_config("navigation")["navigation"]
    runtime_config = load_config("runtime")["runtime"]
    estimator = ProportionalNavigation(nav_config)
    tracker = VisionTracker(root / model_config["engine"], device=runtime_config["device"])
    source = VideoSource(args.video)
    window = "PN test | Q / Esc: quit"
    window_created = False
    processed = detected = predicted = 0
    try:
        if args.show:
            from scripts.test_detection_and_tracking import configure_preview_qt

            configure_preview_qt()
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            window_created = True
            cv2.resizeWindow(window, 960, 540)
        print(f"Video: {args.video} | Source FPS: {source.fps:.3f}"
              f"{' (estimated)' if source.fps_is_estimated else ''} | model: {model_config['engine']}",
              flush=True)
        print("Estimated results; LOS relative to the camera, simplified PN, conditional TTC.", flush=True)
        while not args.max_frames or processed < args.max_frames:
            ok, frame, dt, frame_id = source.read_frame()
            if not ok:
                break
            height, width = frame.shape[:2]
            if (width, height) != estimator.image_size:
                raise ValueError(f"Expected frame size {estimator.image_size}, got {(width, height)}")
            tracking = tracker.process_frame(frame)
            result = estimator.process(tracking, dt, frame_id, (width, height), source.fps_is_estimated)
            processed += 1
            detected += int(tracking.detected and tracking.state is not None)
            predicted += int(not tracking.detected and tracking.state is not None)
            if (frame_id - 1) % args.every == 0:
                print(format_result(frame_id, dt, tracking, result), flush=True)

            if args.show:
                preview = cv2.resize(frame, (960, 540))
                if tracking.state is not None:
                    x, y, w, h = tracking.state[:4]
                    p1 = (round((x - w / 2) * 960 / width), round((y - h / 2) * 540 / height))
                    p2 = (round((x + w / 2) * 960 / width), round((y + h / 2) * 540 / height))
                    color = (0, 255, 0) if tracking.detected else (0, 165, 255)
                    cv2.rectangle(preview, p1, p2, color, 2)
                    label = f"{'Detection' if tracking.detected else 'Prediction'} ID={tracking.target_id}"
                else:
                    label = "No state"
                cv2.putText(preview, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow(window, preview)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        source.close()
        if window_created:
            cv2.destroyWindow(window)
    print(f"Processed {processed} frames: detected={detected}, predicted={predicted}, "
          f"no state={processed - detected - predicted}.")


if __name__ == "__main__":
    main()
