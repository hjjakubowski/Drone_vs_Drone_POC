"""Tracking test with a preview. Press Q or Esc to quit."""

import argparse
import os
from pathlib import Path
import subprocess

import cv2

from project.config import load_config, project_root
from project.video.video_source import VideoSource
from project.vision.tracker_engine import VisionTracker


def configure_preview_qt() -> None:
    """Find system fonts if the OpenCV Qt font directory is missing on Linux."""
    font_dir = os.environ.get("QT_QPA_FONTDIR")
    if font_dir and not Path(font_dir).is_dir():
        try:
            font = subprocess.run(
                ["fc-match", "-f", "%{file}", "sans-serif"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            font = ""
        if font and Path(font).is_file():
            os.environ["QT_QPA_FONTDIR"] = str(Path(font).parent)

    # OpenCV Qt 5 uses XWayland by default in a GNOME session.
    if (os.environ.get("XDG_SESSION_TYPE") == "wayland"
            and os.environ.get("DISPLAY")
            and not os.environ.get("QT_QPA_PLATFORM")):
        os.environ["QT_QPA_PLATFORM"] = "xcb"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true", help="Disable the preview window")
    parser.add_argument("--max-frames", type=int, default=0, help="Frame limit; 0 = entire video")
    args = parser.parse_args()
    if args.max_frames < 0:
        parser.error("--max-frames must not be negative")

    root = project_root()
    config = load_config("model")["model"]
    tracker = VisionTracker(root / config["engine"])
    source = VideoSource(root / "datasets/recordings/fpv_race_test.mp4")
    window = "Tracking preview | Q / Esc: quit"

    try:
        if not args.no_show:
            configure_preview_qt()
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, 960, 540)

        while not args.max_frames or source.frame_id < args.max_frames:
            ok, frame, dt, frame_id = source.read_frame()
            if not ok:
                break

            result = tracker.process_frame(frame)
            if result.state is not None:
                xc, yc, w, h = result.state[:4]
                status = "detection" if result.detected else f"prediction {result.missing_frames}"
                print(f"Frame {frame_id}: {status}, ID={result.target_id}, center=({xc:.1f}, {yc:.1f})")
            else:
                print(f"Frame {frame_id}: no current state, ID={result.target_id}")

            if not args.no_show:
                height, width = frame.shape[:2]
                scale = min(960 / width, 540 / height, 1.0)
                preview = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame.copy()
                if result.state is not None:
                    x1, y1 = round((xc - w / 2) * scale), round((yc - h / 2) * scale)
                    x2, y2 = round((xc + w / 2) * scale), round((yc + h / 2) * scale)
                    color = (0, 255, 0) if result.detected else (0, 165, 255)
                    cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
                    status = "Detection" if result.detected else f"Prediction {result.missing_frames}"
                    label = f"{status} | ID: {result.target_id}"
                elif result.target_id is not None:
                    label = f"No detection | ID: {result.target_id}"
                else:
                    label = "No target"
                cv2.putText(preview, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow(window, preview)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        source.close()
        if not args.no_show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
