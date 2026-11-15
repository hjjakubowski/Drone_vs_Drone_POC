import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

from project.config import load_config, project_root
from project.video.video_source import VideoSource
from project.vision.engine_metadata import read_engine_metadata


def main():

    """Saving detection results from model.enging into CSV file"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Path to the video")
    parser.add_argument("--output", type=Path, default=Path("detection_confidence.csv"))
    parser.add_argument("--conf", type=float, default=0.01, help="Minimum detector confidence")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = entire video")
    args = parser.parse_args()
    if not 0 < args.conf <= 1 or args.max_frames < 0:
        parser.error("conf must be in (0, 1] and max-frames must not be negative")
    root = project_root()
    config = load_config("model")["model"]
    engine = root / config["engine"]
    model = YOLO(str(engine), task="detect")
    imgsz = read_engine_metadata(engine)["imgsz"]
    source = VideoSource(args.video)
    count = 0
    try:
        with args.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["frame_id", "class_id", "confidence", "x1", "y1", "x2", "y2"])
            while not args.max_frames or source.frame_id < args.max_frames:
                ok, frame, _, frame_id = source.read_frame()
                if not ok:
                    break
                result = model.predict(source=frame, imgsz=imgsz, rect=config["rect"],
                                       quantize=config["quantize"], classes=config["classes"],
                                       conf=args.conf, device="cuda:0", verbose=False)[0]
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    writer.writerow([frame_id, "", "", "", "", "", ""])
                    continue
                for box in boxes:
                    coords = box.xyxy[0].cpu().tolist()
                    writer.writerow([frame_id, int(box.cls.item()), float(box.conf.item()), *coords])
                    count += 1
    finally:
        source.close()
    print(f"Saved {count} detections from {source.frame_id} frames to {args.output}")


if __name__ == "__main__":
    main()
