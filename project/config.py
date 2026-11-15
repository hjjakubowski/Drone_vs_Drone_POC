"""Shared paths and YAML configuration, independent of the desktop frontend."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml


def project_root() -> Path:
    override = os.environ.get("INZ_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def resource_path(relative_path: str) -> Path:
    return Path(getattr(sys, "_MEIPASS", project_root())) / relative_path


def load_config(name: str) -> dict:
    directory = Path(os.environ.get("INZ_CONFIG_DIR", project_root() / "configs"))
    if not directory.is_absolute():
        directory = project_root() / directory
    with (directory / f"{name}.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid configuration: {name}.yaml")
    return config


def recordings_dir() -> Path:
    override = os.environ.get("INZ_RECORDINGS_DIR")
    if override:
        return Path(override).expanduser().resolve()
    directory = Path(load_config("runtime")["runtime"]["video_source_dir"])
    return directory if directory.is_absolute() else project_root() / directory
