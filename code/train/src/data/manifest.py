from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .types import SampleRecord


REQUIRED_COLUMNS = {"relative_path", "label", "split"}
VALID_SPLITS = {"train", "val", "test"}


def _read_config(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None

    normalized = value.strip().lower()
    if normalized == "":
        return None
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"Unable to parse boolean value: {value!r}")


def _parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    stripped = value.strip()
    if stripped == "":
        return None
    return int(stripped)


def _normalize_path(value: str) -> str:
    return value.replace("\\", "/").strip()


def _ensure_relative_to_root(root: Path, candidate: Path) -> None:
    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve()
    if not resolved_candidate.is_relative_to(resolved_root):
        raise ValueError(
            f"Image path {resolved_candidate} escapes dataset root {resolved_root}."
        )


def _read_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Manifest is empty: {manifest_path}")

        columns = [column.strip() for column in header]
        missing = REQUIRED_COLUMNS.difference(columns)
        if missing:
            raise ValueError(
                f"Manifest {manifest_path} is missing required columns: {sorted(missing)}"
            )

        rows: list[dict[str, str]] = []
        for row_index, raw_row in enumerate(reader, start=2):
            if not raw_row or all(cell.strip() == "" for cell in raw_row):
                continue

            normalized_row: dict[str, str] = {}
            for index, column in enumerate(columns):
                normalized_row[column] = raw_row[index].strip() if index < len(raw_row) else ""
            normalized_row["_row_number"] = str(row_index)
            rows.append(normalized_row)

    return rows


def load_manifest(dataset_cfg: Any, split: str) -> list[SampleRecord]:
    normalized_split = split.strip().lower()
    if normalized_split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}, got {split!r}")

    dataset_root_value = _read_config(dataset_cfg, "dataset_root", None)
    manifest_path_value = _read_config(dataset_cfg, "manifest_path", None)
    label_map = _read_config(dataset_cfg, "label_map", {})

    if dataset_root_value in {None, ""}:
        raise ValueError("dataset_root must be provided.")
    if manifest_path_value in {None, ""}:
        raise ValueError("manifest_path must be provided.")
    dataset_root = Path(str(dataset_root_value))
    manifest_path = Path(str(manifest_path_value))
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root does not exist: {dataset_root}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"dataset_root is not a directory: {dataset_root}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest_path does not exist: {manifest_path}")
    if not isinstance(label_map, Mapping) or not label_map:
        raise ValueError("label_map must be a non-empty mapping.")

    samples: list[SampleRecord] = []
    for row in _read_manifest_rows(manifest_path):
        row_number = row["_row_number"]
        row_split = row["split"].strip().lower()
        if row_split not in VALID_SPLITS:
            raise ValueError(
                f"Manifest row {row_number} has unsupported split {row['split']!r}."
            )

        label_name = row["label"].strip()
        if label_name not in label_map:
            raise ValueError(
                f"Manifest row {row_number} has unknown label {label_name!r}; "
                f"expected one of {sorted(label_map)}."
            )

        relative_path = _normalize_path(row["relative_path"])
        if not relative_path:
            raise ValueError(f"Manifest row {row_number} has an empty relative_path.")

        image_path = dataset_root / Path(relative_path)
        _ensure_relative_to_root(dataset_root, image_path)
        if not image_path.exists():
            raise FileNotFoundError(
                f"Manifest row {row_number} points to a missing file: {image_path}"
            )

        if row_split != normalized_split:
            continue

        samples.append(
            SampleRecord(
                relative_path=relative_path,
                image_path=image_path,
                label_name=label_name,
                label_id=int(label_map[label_name]),
                split=row_split,
                width=_parse_optional_int(row.get("width")),
                height=_parse_optional_int(row.get("height")),
                size_bucket=row.get("size_bucket") or None,
                aspect_bucket=row.get("aspect_bucket") or None,
                difficulty_tag=row.get("difficulty_tag") or None,
                was_converted_to_rgb=_parse_bool(row.get("was_converted_to_rgb")),
            )
        )

    if not samples:
        raise ValueError(
            f"No samples found for split {normalized_split!r} in manifest {manifest_path}."
        )

    return samples
