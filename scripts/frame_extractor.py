import argparse
from pathlib import Path
from secrets import randbelow

import cv2


FRAME_INTERVAL = 15


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract frames from an MP4 video every 0.5 seconds."
    )
    parser.add_argument(
        "--video_path",
        required=True,
        help="Path to the input MP4 video.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where JPG frames will be saved.",
    )

    return parser.parse_args()


def validate_video_path(video_path):
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    if video_path.suffix.lower() != ".mp4":
        raise ValueError("Input video must be an MP4 file.")


def generate_unique_frame_path(output_dir):
    while True:
        filename = f"{randbelow(100_000_000):08d}.jpg"
        frame_path = output_dir / filename

        if not frame_path.exists():
            return frame_path


def frame_capturer(video_path, output_dir):
    video = cv2.VideoCapture(str(video_path))

    try:
        if not video.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        frame_number = 0
        saved_frames = 0

        while True:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, frame = video.read()

            if not success:
                break

            frame_path = generate_unique_frame_path(output_dir)
            saved = cv2.imwrite(str(frame_path), frame)

            if not saved:
                raise RuntimeError(f"Could not save frame: {frame_path}")

            saved_frames += 1
            frame_number += FRAME_INTERVAL

        return saved_frames

    finally:
        video.release()


def main():
    args = parse_args()

    video_path = Path(args.video_path)
    output_dir = Path(args.output_dir)

    validate_video_path(video_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_frames = frame_capturer(video_path, output_dir)
    print(f"Saved {saved_frames} frames to: {output_dir}")


if __name__ == "__main__":
    main()
