from __future__ import annotations

import os
import sys
from pathlib import Path


def project_root() -> Path:
    override = os.environ.get("INZ_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent

    return Path(__file__).resolve().parents[3]


def resource_path(relative_path: str) -> Path:
    base_path = Path(getattr(sys, "_MEIPASS", project_root()))
    return base_path / relative_path


def recordings_dir() -> Path:
    override = os.environ.get("INZ_RECORDINGS_DIR")
    if override:
        return Path(override).expanduser().resolve()

    root = project_root()
    current_location = root / "datasets" / "recordings"
    legacy_location = root / "recordings"
    if current_location.exists():
        return current_location
    return legacy_location
