from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.hud_desktop.utils.paths import recordings_dir


@dataclass(frozen=True)
class ViewDefinition:
    id: str
    name: str
    video_path: Path
    description: str
    status: str = "READY"


def load_view_definitions() -> list[ViewDefinition]:
    videos = sorted(recordings_dir().glob("*.mp4"))
    fallback_names = [
        "20260509_113340.mp4",
        "20260509_113727.mp4",
        "20260509_113835.mp4",
    ]

    if not videos:
        videos = [recordings_dir() / name for name in fallback_names]

    definitions: list[ViewDefinition] = []
    for index in range(4):
        video_path = videos[index % len(videos)]
        definitions.append(
            ViewDefinition(
                id=f"view-{index + 1}",
                name=f"Widok-{index + 1}",
                video_path=video_path,
                description=_description_for(index),
                status="READY" if video_path.exists() else "NO VIDEO",
            )
        )
    return definitions


def _description_for(index: int) -> str:
    descriptions = [
        "Pościg.",
        "Przechwycenie od boku.",
        "Przechwycenie od przodu.",
        "Przechwycenie od dołu.",
    ]
    return descriptions[index]
