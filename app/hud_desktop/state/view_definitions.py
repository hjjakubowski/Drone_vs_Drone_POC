from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from project.config import recordings_dir


@dataclass(frozen=True)
class ViewDefinition:
    id: str
    name: str
    video_path: Path
    description: str
    status: str = "READY"


def load_view_definitions() -> list[ViewDefinition]:
    directory = recordings_dir()
    videos = sorted(p for p in directory.glob("*") if p.is_file() and p.suffix.lower() == ".mp4")
    fallback_names = [
        "20260509_113340.mp4",
        "20260509_113727.mp4",
        "20260509_113835.mp4",
    ]

    if not videos:
        videos = [directory / name for name in fallback_names]

    definitions: list[ViewDefinition] = []
    for index in range(4):
        # Preserve the four demo slots even when fewer recordings are available.
        video_path = videos[index % len(videos)]
        definitions.append(
            ViewDefinition(
                id=f"view-{index + 1}",
                name=f"View-{index + 1}",
                video_path=video_path,
                description=_description_for(index),
                status="READY" if video_path.exists() else "NO VIDEO",
            )
        )
    return definitions


def _description_for(index: int) -> str:
    descriptions = [
        "Pursuit.",
        "Side approach.",
        "Front approach.",
        "Approach from below.",
    ]
    return descriptions[index]
