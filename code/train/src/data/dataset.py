from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from ..utils.config import load_experiment_config


@dataclass(frozen=True)
class SampleRecord:
    relative_path: str
    label: str
    split: str
    width: int | None = None
    height: int | None = None
    size_bucket: str | None = None
    aspect_bucket: str | None = None
    difficulty_tag: str | None = None
    was_converted_to_rgb: bool | None = None


def build_label_map(labels: Sequence[str] | dict[str, int]) -> dict[str, int]:
    """Build a stable label map from either a label list or an explicit mapping."""
    if isinstance(labels, dict):
        return {str(label).strip(): int(index) for label, index in labels.items()}

    normalized_labels = sorted({str(label).strip() for label in labels if str(label).strip()})
    return {label: index for index, label in enumerate(normalized_labels)}


def _clean_manifest_row(row: dict[str, Any]) -> dict[str, str]:
    return {str(key).strip(): str(value).strip() for key, value in row.items() if key is not None}


def _optional_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _optional_bool(value: str | None) -> bool | None:
    if value in (None, ""):
        return None
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot parse boolean value from manifest: {value}")


def load_manifest(manifest_path: str | Path, split: str | None = None) -> list[SampleRecord]:
    """Load manifest.csv and optionally keep only one split."""
    manifest = Path(manifest_path).expanduser().resolve()
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest file does not exist: {manifest}")

    records: list[SampleRecord] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        for raw_row in reader:
            row = _clean_manifest_row(raw_row)
            record = SampleRecord(
                relative_path=row["relative_path"],
                label=row["label"],
                split=row["split"],
                width=_optional_int(row.get("width")),
                height=_optional_int(row.get("height")),
                size_bucket=row.get("size_bucket") or None,
                aspect_bucket=row.get("aspect_bucket") or None,
                difficulty_tag=row.get("difficulty_tag") or None,
                was_converted_to_rgb=_optional_bool(row.get("was_converted_to_rgb")),
            )
            if split is None or record.split == split:
                records.append(record)

    if split is not None and not records:
        raise ValueError(f"No samples found for split '{split}' in manifest: {manifest}")
    return records


def build_transforms(stage: str, image_size: int, normalize_cfg: dict[str, Any]) -> list[Any]:
    """Create MindSpore image transforms for the requested data stage."""
    import mindspore.dataset.vision as vision

    if stage not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported stage: {stage}")

    if "mean" not in normalize_cfg or "std" not in normalize_cfg:
        raise KeyError("normalize config must contain both 'mean' and 'std'")

    resize_size = max(int(round(image_size / 0.875)), image_size)
    common_ops: list[Any] = [
        vision.Resize((resize_size, resize_size), interpolation=vision.Inter.BILINEAR),
    ]

    if stage == "train":
        common_ops.extend(
            [
                vision.RandomCrop((image_size, image_size)),
                vision.RandomHorizontalFlip(prob=0.5),
            ]
        )
    else:
        common_ops.append(vision.CenterCrop((image_size, image_size)))

    common_ops.extend(
        [
            vision.Rescale(1.0 / 255.0, 0.0),
            vision.Normalize(mean=normalize_cfg["mean"], std=normalize_cfg["std"]),
            vision.HWC2CHW(),
        ]
    )
    return common_ops


class ManifestImageDataset:
    """Simple manifest-backed dataset that loads RGB images on demand."""

    def __init__(
        self,
        records: Sequence[SampleRecord],
        dataset_root: str | Path,
        label_map: dict[str, int],
    ) -> None:
        self.records = list(records)
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.label_map = build_label_map(label_map)

        missing_labels = sorted({record.label for record in self.records if record.label not in self.label_map})
        if missing_labels:
            raise KeyError(f"Labels missing from label_map: {missing_labels}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.int32]:
        record = self.records[index]
        image_path = self.dataset_root / Path(record.relative_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_array = np.asarray(image)

        label_id = np.int32(self.label_map[record.label])
        return image_array, label_id


def _resolve_data_path(path_value: str | Path, config: dict[str, Any]) -> Path:
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    project_root = Path(config.get("project_root", Path.cwd())).expanduser().resolve()
    project_candidate = (project_root / candidate).resolve()
    if project_candidate.exists():
        return project_candidate
    return (Path.cwd() / candidate).resolve()


def build_dataset(
    records: Sequence[SampleRecord],
    dataset_root: str | Path,
    stage: str,
    config: dict[str, Any],
):
    """Build a MindSpore GeneratorDataset from manifest records."""
    import mindspore.dataset as ds
    import mindspore.dataset.transforms as transforms

    dataset_cfg = config["dataset"]
    runtime_cfg = config["runtime"]
    label_map = build_label_map(dataset_cfg["label_map"])
    resolved_dataset_root = _resolve_data_path(dataset_root, config)
    generator = ManifestImageDataset(records=records, dataset_root=resolved_dataset_root, label_map=label_map)

    dataset = ds.GeneratorDataset(
        source=generator,
        column_names=["image", "label"],
        shuffle=stage == "train",
        num_parallel_workers=runtime_cfg.get("num_parallel_workers", 1),
    )
    dataset = dataset.map(
        operations=build_transforms(
            stage=stage,
            image_size=int(dataset_cfg["image_size"]),
            normalize_cfg=dataset_cfg["normalize"],
        ),
        input_columns=["image"],
        num_parallel_workers=runtime_cfg.get("num_parallel_workers", 1),
    )
    dataset = dataset.map(
        operations=transforms.TypeCast(np.int32),
        input_columns=["label"],
        num_parallel_workers=runtime_cfg.get("num_parallel_workers", 1),
    )
    return dataset


def _normalize_config(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, (str, Path)):
        return load_experiment_config(config)
    return config


def build_dataloader(split: str, config: str | Path | dict[str, Any]):
    """Build a split-specific MindSpore dataloader from experiment config."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}")

    normalized_config = _normalize_config(config)
    dataset_cfg = normalized_config["dataset"]
    train_cfg = normalized_config["train"]

    manifest_path = _resolve_data_path(dataset_cfg["manifest_path"], normalized_config)
    records = load_manifest(manifest_path, split=split)
    dataset = build_dataset(
        records=records,
        dataset_root=dataset_cfg["dataset_root"],
        stage=split,
        config=normalized_config,
    )
    return dataset.batch(
        batch_size=int(train_cfg["batch_size"]),
        drop_remainder=split == "train",
    )


__all__ = [
    "ManifestImageDataset",
    "SampleRecord",
    "build_dataloader",
    "build_dataset",
    "build_label_map",
    "build_transforms",
    "load_manifest",
]
