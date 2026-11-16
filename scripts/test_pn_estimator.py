"""Video analysis: VideoSource → VisionTracker → ProportionalNavigation."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2

from project.config import load_config, project_root
from project.navigation.pn_estimator import ProportionalNavigation
from project.video.video_source import VideoSource
from project.vision.tracker_engine import VisionTracker


class SmoothingComparison:
    """Comparing estimators on identical frames."""

    def __init__(self, config: dict, fps: float) -> None:
        windows = {
            "baseline": config["closing_speed_window"],
            **{f"{int(seconds * 1000)}ms": max(1, round(seconds * fps))
               for seconds in (0.1, 0.2, 0.3)},
        }
        self.windows = windows
        self.estimators = {
            name: ProportionalNavigation({**config, "closing_speed_window": size})
            for name, size in windows.items()
        }
        self.derivatives = 0

    def process(self, tracking, dt, frame_id, image_size, fps_is_estimated):
        results = {
            name: estimator.process(tracking, dt, frame_id, image_size, fps_is_estimated)
            for name, estimator in self.estimators.items()
        }
        # A missing derivative indicates an estimator reset or its first detection.
        if results["baseline"].closing_speed is None:
            self.derivatives = 0
        else:
            self.derivatives += 1
        ready = {name: self.derivatives >= size for name, size in self.windows.items()}
        return results, ready


def csv_record(frame_id, dt, tracking, results, ready, previous):
    """Full precison, empty field only for lack of detection in the frame, changes are applied only for frames with detection."""
    state = tracking.state
    status = ("detection" if tracking.detected and state is not None else
              "prediction" if state is not None and not tracking.detected else "no_state")
    row = {
        "frame_id": frame_id, "time_s": (frame_id - 1) * dt, "dt_s": dt,
        "fps_is_estimated": results["baseline"].fps_is_estimated,
        "status": status, "target_id": tracking.target_id, "missing_frames": tracking.missing_frames,
    }
    fields = ("xc_px", "yc_px", "width_px", "height_px", "vx_px_per_frame",
              "vy_px_per_frame", "vw_px_per_frame", "vh_px_per_frame")
    if state is not None and len(state) == 8:
        row.update(zip(fields, map(float, state)))
    else:
        row.update(dict.fromkeys(fields))
    same = (status == "detection" and previous is not None
            and previous["status"] == "detection" and previous["target_id"] == tracking.target_id
            and previous["frame_id"] == frame_id - 1)
    for field in ("xc_px", "yc_px", "width_px", "height_px"):
        row[f"delta_{field}"] = (row[field] - previous[field]
                                  if same and row[field] is not None and previous[field] is not None
                                  else None)
    result = results["baseline"]
    for field in ("los_x", "los_y", "los_rate_x", "los_rate_y", "depth_z", "closing_speed",
                  "ttc", "accel_x", "accel_y"):
        row[field] = getattr(result, field)
    for name, comparison in results.items():
        row[f"speed_{name}_m_s"] = comparison.closing_speed
        row[f"window_{name}_ready"] = ready[name]
        row[f"pn_x_{name}_m_s2"] = comparison.accel_x
        row[f"pn_y_{name}_m_s2"] = comparison.accel_y
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="1920×1080 video containing a drone")
    parser.add_argument("--max-frames", type=int, default=0, help="Frame limit; 0 means the entire video")
    parser.add_argument("--show", action="store_true", help="Show a video preview and the target box")
    args = parser.parse_args()
    if args.max_frames < 0:
        parser.error("--max-frames must not be negative")

    root = project_root()
    model_config = load_config("model")["model"]
    nav_config = load_config("navigation")["navigation"]
    runtime_config = load_config("runtime")["runtime"]
    tracker = VisionTracker(root / model_config["engine"], device=runtime_config["device"])
    source = VideoSource(args.video)
    try:
        comparison = SmoothingComparison(nav_config, source.fps)
    except Exception:
        source.close()
        raise
    estimator = comparison.estimators["baseline"]
    window = "PN test | Q / Esc: quit"
    window_created = False
    processed = 0
    csv_stream = None
    writer = None
    previous_row = None
    try:
        report_dir = root / "research" / "reports" / "telemetry_pn"
        report_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        csv_path = report_dir / f"{args.video.stem}_{stamp}_{uuid4().hex[:8]}.csv"
        csv_stream = csv_path.open("x", encoding="utf-8", newline="")
        if args.show:
            from scripts.test_detection_and_tracking import configure_preview_qt

            configure_preview_qt()
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            window_created = True
            cv2.resizeWindow(window, 960, 540)
        while not args.max_frames or processed < args.max_frames:
            ok, frame, dt, frame_id = source.read_frame()
            if not ok:
                break
            height, width = frame.shape[:2]
            if (width, height) != estimator.image_size:
                raise ValueError(f"Expected frame size {estimator.image_size}, got {(width, height)}")
            tracking = tracker.process_frame(frame)
            results, ready = comparison.process(tracking, dt, frame_id, (width, height),
                                                source.fps_is_estimated)
            row = csv_record(frame_id, dt, tracking, results, ready, previous_row)
            row.update(source_fps=source.fps, fx_px=nav_config["fx"], fy_px=nav_config["fy"],
                       target_width_m=nav_config["target_width_m"], guidance_gain=nav_config["guidance_gain"])
            previous_row = row
            if writer is None:
                writer = csv.DictWriter(csv_stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            csv_stream.flush()
            processed += 1

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
        if csv_stream is not None:
            csv_stream.close()
        if window_created:
            cv2.destroyWindow(window)


if __name__ == "__main__":
    main()
