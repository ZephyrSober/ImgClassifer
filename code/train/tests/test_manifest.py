from __future__ import annotations

import csv
import shutil
import sys
import unittest
import uuid
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAIN_SRC = PROJECT_ROOT / "code" / "train" / "src"
TEST_TMP_ROOT = PROJECT_ROOT / "code" / "train" / "tests" / "_tmp"
if str(TRAIN_SRC) not in sys.path:
    sys.path.insert(0, str(TRAIN_SRC))

from data.manifest import load_manifest  # noqa: E402


def _make_image(path: Path, size: tuple[int, int] = (256, 256)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(120, 80, 30)).save(path)


def _write_manifest(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class LoadManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.root = TEST_TMP_ROOT / f"{self._testMethodName}_{uuid.uuid4().hex}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.dataset_root = self.root / "cleaned"
        self.manifest_path = self.root / "manifest.csv"
        _make_image(self.dataset_root / "Cat" / "cat_0.jpg")
        _make_image(self.dataset_root / "Cat" / "cat_1.jpg")
        _make_image(self.dataset_root / "Dog" / "dog_0.jpg")

        self.base_rows = [
            {
                "relative_path": "Cat/cat_0.jpg",
                "label": "Cat",
                "split": "train",
                "width": "256",
                "height": "256",
                "size_bucket": "medium",
                "aspect_bucket": "balanced",
                "difficulty_tag": "standard",
                "was_converted_to_rgb": "False",
            },
            {
                "relative_path": "Cat/cat_1.jpg",
                "label": "Cat",
                "split": "val",
                "width": "256",
                "height": "256",
                "size_bucket": "medium",
                "aspect_bucket": "balanced",
                "difficulty_tag": "standard",
                "was_converted_to_rgb": "False",
            },
            {
                "relative_path": "Dog/dog_0.jpg",
                "label": "Dog",
                "split": "test",
                "width": "256",
                "height": "256",
                "size_bucket": "medium",
                "aspect_bucket": "balanced",
                "difficulty_tag": "standard",
                "was_converted_to_rgb": "True",
            },
        ]
        self.fieldnames = [
            "relative_path",
            "label",
            "split",
            "width",
            "height",
            "size_bucket",
            "aspect_bucket",
            "difficulty_tag",
            "was_converted_to_rgb",
        ]

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _dataset_cfg(self) -> dict[str, object]:
        return {
            "dataset_root": str(self.dataset_root),
            "manifest_path": str(self.manifest_path),
            "label_map": {"Cat": 0, "Dog": 1},
            "image_size": 224,
            "normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        }

    def test_load_manifest_filters_split_and_maps_labels(self) -> None:
        _write_manifest(self.manifest_path, self.base_rows, self.fieldnames)

        samples = load_manifest(self._dataset_cfg(), "train")

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].relative_path, "Cat/cat_0.jpg")
        self.assertEqual(samples[0].label_name, "Cat")
        self.assertEqual(samples[0].label_id, 0)
        self.assertFalse(samples[0].was_converted_to_rgb)

    def test_load_manifest_fails_when_required_column_is_missing(self) -> None:
        fieldnames = ["relative_path", "label"]
        _write_manifest(
            self.manifest_path,
            [{"relative_path": "Cat/cat_0.jpg", "label": "Cat"}],
            fieldnames,
        )

        with self.assertRaisesRegex(ValueError, "missing required columns"):
            load_manifest(self._dataset_cfg(), "train")

    def test_load_manifest_fails_for_missing_image_path(self) -> None:
        rows = list(self.base_rows)
        rows[0] = {**rows[0], "relative_path": "Cat/missing.jpg"}
        _write_manifest(self.manifest_path, rows, self.fieldnames)

        with self.assertRaisesRegex(FileNotFoundError, "missing file"):
            load_manifest(self._dataset_cfg(), "train")

    def test_load_manifest_fails_for_unknown_label(self) -> None:
        rows = list(self.base_rows)
        rows[0] = {**rows[0], "label": "Bird"}
        _write_manifest(self.manifest_path, rows, self.fieldnames)

        with self.assertRaisesRegex(ValueError, "unknown label"):
            load_manifest(self._dataset_cfg(), "train")

    def test_load_manifest_fails_when_split_is_empty(self) -> None:
        rows = [{**row, "split": "val"} for row in self.base_rows]
        _write_manifest(self.manifest_path, rows, self.fieldnames)

        with self.assertRaisesRegex(ValueError, "No samples found for split"):
            load_manifest(self._dataset_cfg(), "train")
