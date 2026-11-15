from pathlib import Path
from ultralytics import YOLO
from project.config import load_config, project_root


def main():
    """ Exporting Pytorch model into TensorRT format"""

    config = load_config("model")
    root = project_root()
    destination = root / config["model"]["engine"]
    if destination.exists():
        raise FileExistsError(f"Destination file already exists: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(root / config["model"]["weights"])
    Path(model.export(**config["export"])).rename(destination)
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
