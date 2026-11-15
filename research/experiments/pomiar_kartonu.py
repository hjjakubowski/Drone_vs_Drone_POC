"""Manually measure cardboard-box corners in video frames.

Point order: top left, top right, bottom right, bottom left.
Results are saved incrementally to a new output directory.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np


POINT_NAMES = ("lewy_gorny", "prawy_gorny", "prawy_dolny", "lewy_dolny")
WINDOW = "Cardboard box measurement"


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Value must be positive and finite")
    return number


def frame_indices(total: int, samples: int) -> list[int]:
    """Evenly spaced samples within the video; indices start at zero."""
    if total <= 0 or not 1 <= samples <= total:
        raise ValueError("Sample count must be between one and the frame count")
    return [int((i + 0.5) * total / samples) for i in range(samples)]


def select_frame_indices(total: int, fps: float, *, samples: int | None = None,
                         frames: list[int] | None = None,
                         times: list[float] | None = None) -> list[int]:
    """Select zero-based indices; times assume constant FPS and a start at t=0."""
    if total <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("The video must have a valid frame count and FPS")
    if sum(value is not None for value in (samples, frames, times)) > 1:
        raise ValueError("Choose only one frame-selection method")
    if frames is not None:
        if not frames or any(index < 0 or index >= total for index in frames):
            raise ValueError(f"Frame indices must be in the range 0–{total - 1}")
        selected = frames
    elif times is not None:
        duration = total / fps
        if not times or any(not math.isfinite(t) or t < 0 or t >= duration for t in times):
            raise ValueError(f"Times must be in the range [0, {duration:.3f}) s")
        # Round to the nearest frame and clamp to the last frame.
        selected = [min(total - 1, math.floor(t * fps + 0.5)) for t in times]
    else:
        return frame_indices(total, 5 if samples is None else samples)
    return sorted(set(selected))


def measure(points: list[tuple[float, float]], distance_m: float, width_m: float) -> dict:
    if len(points) != 4:
        raise ValueError("Four corners are required")
    p = np.asarray(points, dtype=float)
    if not np.isfinite(p).all():
        raise ValueError("Coordinates must be finite")
    # Euclidean image distances, not a bounding rectangle.
    top, right, bottom, left = (
        float(np.linalg.norm(p[(i + 1) % 4] - p[i])) for i in range(4)
    )
    if min(top, right, bottom, left) <= 0:
        raise ValueError("Corners must not form zero-length edges")
    width = (top + bottom) / 2
    return {
        "gorna_px": top, "prawa_px": right, "dolna_px": bottom, "lewa_px": left,
        "szerokosc_srednia_px": width, "wysokosc_srednia_px": (left + right) / 2,
        "fx_przyblizone_px": width * distance_m / width_m,
    }


def configure_qt() -> None:
    """OpenCV wheels may point to a missing Qt font directory."""
    font_dir = os.environ.get("QT_QPA_FONTDIR")
    if font_dir and not Path(font_dir).is_dir():
        try:
            font = subprocess.run(
                ["fc-match", "-f", "%{file}", "sans-serif"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            if font and Path(font).is_file():
                os.environ["QT_QPA_FONTDIR"] = str(Path(font).parent)
        except (OSError, subprocess.CalledProcessError):
            pass
    if (os.environ.get("XDG_SESSION_TYPE") == "wayland"
            and os.environ.get("DISPLAY") and not os.environ.get("QT_QPA_PLATFORM")):
        os.environ["QT_QPA_PLATFORM"] = "xcb"


def select_points(frame: np.ndarray, index: int, ordinal: int, total: int,
                  preview_width: int) -> list[tuple[float, float]] | None | str:
    height, width = frame.shape[:2]
    display_width = min(preview_width, width)
    display_height = max(1, round(height * display_width / width))
    sx, sy = width / display_width, height / display_height
    base = cv2.resize(frame, (display_width, display_height))
    clicks: list[tuple[int, int]] = []

    def on_click(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((x, y))

    cv2.setMouseCallback(WINDOW, on_click)
    while True:
        display = base.copy()
        for i, point in enumerate(clicks):
            cv2.circle(display, point, 5, (0, 255, 255), -1)
            cv2.putText(display, str(i + 1), (point[0] + 7, point[1] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if i:
                cv2.line(display, clicks[i - 1], point, (0, 255, 0), 2)
        if len(clicks) == 4:
            cv2.line(display, clicks[-1], clicks[0], (0, 255, 0), 2)
        text = f"Frame {index} ({ordinal}/{total}) | 1:TL 2:TR 3:BR 4:BL"
        cv2.rectangle(display, (0, 0), (display_width, 54), (0, 0, 0), -1)
        cv2.putText(display, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        cv2.putText(display, "Enter: save | U: undo | R: reset | S: skip | Q/Esc: quit",
                    (8, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(30) & 0xFF
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1 or key in (ord("q"), 27):
            return "quit"
        if key == ord("s"):
            return None
        if key == ord("u") and clicks:
            clicks.pop()
        if key == ord("r"):
            clicks.clear()
        if key in (10, 13) and len(clicks) == 4:
            points = [(x * sx, y * sy) for x, y in clicks]
            try:
                measure(points, 1.0, 1.0)
            except ValueError as exc:
                print(f"Invalid points: {exc}", flush=True)
                continue
            return points


def annotated_frame(frame: np.ndarray, points: list[tuple[float, float]]) -> np.ndarray:
    annotated = frame.copy()
    pts = np.rint(points).astype(np.int32)
    cv2.polylines(annotated, [pts], True, (0, 255, 0), 3, cv2.LINE_AA)
    for i, (x, y) in enumerate(pts, 1):
        cv2.circle(annotated, (int(x), int(y)), 7, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(annotated, str(i), (int(x) + 10, max(22, int(y) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return annotated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Path to the video")
    parser.add_argument("--distance-m", type=positive_float, required=True, help="Distance from lens to box [m]")
    parser.add_argument("--width-m", type=positive_float, default=0.38, help="Box width [m], default: 0.38")
    parser.add_argument("--height-m", type=positive_float, default=0.18, help="Box height [m], default: 0.18")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--samples", type=int, help="Number of evenly spaced frames (default: 5)")
    selection.add_argument("--frames", type=int, nargs="+", metavar="INDEX",
                           help="Zero-based frame indices, e.g. --frames 0 150 210")
    selection.add_argument("--times", type=float, nargs="+", metavar="SECONDS",
                           help="Times in seconds, e.g. --times 0 5 7 8.5 10")
    parser.add_argument("--preview-width", type=int, default=1280, help="Maximum preview width in pixels")
    parser.add_argument("--output", type=Path, required=True, help="New directory for CSV and images")
    args = parser.parse_args()
    if (args.samples is not None and args.samples < 1) or args.preview_width < 320:
        parser.error("--samples must be positive and --preview-width must be at least 320")
    if not args.video.is_file():
        parser.error(f"Video not found: {args.video}")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        capture.release()
        parser.error(f"Cannot open video: {args.video}")
    window_created = False
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            parser.error("Invalid video FPS; frame times cannot be calculated")
        try:
            indices = select_frame_indices(total, fps, samples=args.samples,
                                           frames=args.frames, times=args.times)
        except ValueError as exc:
            parser.error(str(exc))
        if args.times is not None or args.frames is not None:
            print("Selected frames (zero-based index, time estimated from FPS):", flush=True)
            for index in indices:
                print(f"  {index}: {index / fps:.3f} s", flush=True)
            if len(indices) < len(args.times if args.times is not None else args.frames):
                print("Duplicate frames will be annotated only once.", flush=True)
        try:
            args.output.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            parser.error(f"Output directory already exists: {args.output}; choose a new one")

        configure_qt()
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        window_created = True
        fields = ["video", "frame_index_0", "czas_szacowany_s", "fps", "image_width_px", "image_height_px",
                  "distance_m", "target_width_m", "target_height_m"]
        fields += [f"{name}_{axis}" for name in POINT_NAMES for axis in ("x", "y")]
        fields += ["gorna_px", "prawa_px", "dolna_px", "lewa_px", "szerokosc_srednia_px",
                   "wysokosc_srednia_px", "fx_przyblizone_px", "obraz"]
        csv_path = args.output / "pomiary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            stream.flush()
            for ordinal, index in enumerate(indices, 1):
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if not ok:
                    print(f"Could not read frame {index}; skipped.")
                    continue
                points = select_points(frame, index, ordinal, len(indices), args.preview_width)
                if points == "quit":
                    break
                if points is None:
                    continue
                values = measure(points, args.distance_m, args.width_m)
                filename = f"klatka_{index:06d}_oznaczona.png"
                if not cv2.imwrite(str(args.output / filename), annotated_frame(frame, points)):
                    raise OSError(f"Could not save image: {filename}")
                row = {"video": str(args.video.resolve()), "frame_index_0": index,
                       "czas_szacowany_s": index / fps, "fps": fps,
                       "image_width_px": frame.shape[1], "image_height_px": frame.shape[0],
                       "distance_m": args.distance_m, "target_width_m": args.width_m,
                       "target_height_m": args.height_m, "obraz": filename, **values}
                for name, (x, y) in zip(POINT_NAMES, points):
                    row[f"{name}_x"], row[f"{name}_y"] = x, y
                writer.writerow(row)
                stream.flush()
                print(f"Saved frame {index}, width {values['szerokosc_srednia_px']:.1f} px, "
                      f"approximate fx {values['fx_przyblizone_px']:.1f} px", flush=True)
        print(f"Results: {csv_path}")
    finally:
        capture.release()
        if window_created:
            cv2.destroyWindow(WINDOW)


if __name__ == "__main__":
    main()
