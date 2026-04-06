from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SampleRecord:
    relative_path: str
    image_path: Path
    label_name: str
    label_id: int
    split: str
    width: int | None = None
    height: int | None = None
    size_bucket: str | None = None
    aspect_bucket: str | None = None
    difficulty_tag: str | None = None
    was_converted_to_rgb: bool | None = None
