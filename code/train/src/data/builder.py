from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .manifest import load_manifest
from .transforms import build_transforms
from .types import SampleRecord

try:
    import mindspore.dataset as ds
except ImportError as exc:  # pragma: no cover - exercised only when dependency is missing.
    ds = None
    _MINDSPORE_IMPORT_ERROR = exc
else:
    _MINDSPORE_IMPORT_ERROR = None


def _read_config(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _ensure_mindspore_available() -> None:
    if ds is None:
        raise ImportError(
            "mindspore is required to build dataloaders. Install mindspore before using "
            "code.train.src.data.builder."
        ) from _MINDSPORE_IMPORT_ERROR


class _ImageDataset:
    def __init__(self, samples: list[SampleRecord], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.int32]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            array = self.transform(image)
        return array, np.int32(sample.label_id)


def build_dataloader(dataset_cfg: Any, train_cfg: Any, runtime_cfg: Any, split: str):
    _ensure_mindspore_available()

    normalized_split = split.strip().lower()
    if normalized_split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split!r}")

    batch_size = int(_read_config(train_cfg, "batch_size", 0))
    num_parallel_workers = int(_read_config(runtime_cfg, "num_parallel_workers", 1))
    seed = int(_read_config(runtime_cfg, "seed", 42))

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if num_parallel_workers <= 0:
        raise ValueError(
            f"num_parallel_workers must be positive, got {num_parallel_workers}"
        )

    random.seed(seed)
    np.random.seed(seed)
    ds.config.set_seed(seed)

    samples = load_manifest(dataset_cfg, normalized_split)
    transform = build_transforms(dataset_cfg, normalized_split)
    source = _ImageDataset(samples, transform)

    dataset = ds.GeneratorDataset(
        source=source,
        column_names=["image", "label"],
        shuffle=normalized_split == "train",
        num_parallel_workers=num_parallel_workers,
    )
    return dataset.batch(
        batch_size,
        drop_remainder=normalized_split == "train",
    )


def build_dataloaders(dataset_cfg: Any, train_cfg: Any, runtime_cfg: Any):
    label_map = _read_config(dataset_cfg, "label_map", {})
    if not isinstance(label_map, Mapping) or not label_map:
        raise ValueError("label_map must be a non-empty mapping.")

    dataloaders = {
        split: build_dataloader(dataset_cfg, train_cfg, runtime_cfg, split)
        for split in ("train", "val", "test")
    }
    return dataloaders, len(label_map)
