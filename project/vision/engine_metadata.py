"""Read Ultralytics engine metadata without CUDA or deserializing TensorRT code."""

from __future__ import annotations

import json
from pathlib import Path


def read_engine_metadata(path: Path | str) -> dict:
    with Path(path).open("rb") as stream:
        header = stream.read(4)
        if len(header) != 4:
            raise ValueError("Missing Ultralytics metadata header.")
        size = int.from_bytes(header, "little")
        if not 0 < size <= 1_000_000:
            raise ValueError("Invalid Ultralytics metadata size.")
        raw = stream.read(size)
        if len(raw) != size or not stream.read(1):
            raise ValueError("Incomplete engine file.")
    metadata = json.loads(raw)
    if not isinstance(metadata, dict):
        raise ValueError("Engine metadata is not a JSON object.")
    return metadata


def check_engine_metadata(metadata: dict, export: dict) -> list[str]:
    """Compare declared settings only; this does not validate GPU compatibility."""
    issues = []
    for key in ("batch", "imgsz"):
        if metadata.get(key) != export[key]:
            issues.append(f"{key}: recorded {metadata.get(key)!r}, expected {export[key]!r}")
    arguments = metadata.get("args", {})
    for key in ("quantize", "dynamic", "nms"):
        if arguments.get(key) != export[key]:
            issues.append(f"{key}: recorded {arguments.get(key)!r}, expected {export[key]!r}")
    if export["nms"] is False and metadata.get("end2end") is not True:
        issues.append("end2end: engine does not declare an NMS-free head")
    names = metadata.get("names", {})
    if not isinstance(names, dict) or names.get("2", names.get(2)) != "drone":
        issues.append("classes: drone class is missing at index 2")
    if metadata.get("task") != "detect":
        issues.append("task: expected detect")
    return issues
