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

try:
    import mindspore  # noqa: F401
except ImportError:
    MINDSPORE_AVAILABLE = False
else:
    MINDSPORE_AVAILABLE = True

from data.builder import build_dataloader, build_dataloaders  # noqa: E402


def _make_image(path: Path, size: tuple[int, int] = (320, 256)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(64, 96, 128)).save(path)


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["relative_path", "label", "split"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@unittest.skipUnless(MINDSPORE_AVAILABLE, "mindspore is required for dataloader integration tests")
class BuildDataloaderTests(unittest.TestCase):
    def setUp(self) -> None:
        TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.root = TEST_TMP_ROOT / f"{self._testMethodName}_{uuid.uuid4().hex}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.dataset_root = self.root / "cleaned"
        self.manifest_path = self.root / "manifest.csv"

        _make_image(self.dataset_root / "Cat" / "cat_0.jpg")
        _make_image(self.dataset_root / "Cat" / "cat_1.jpg")
        _make_image(self.dataset_root / "Dog" / "dog_0.jpg")
        _make_image(self.dataset_root / "Dog" / "dog_1.jpg")

        rows = [
            {"relative_path": "Cat/cat_0.jpg", "label": "Cat", "split": "train"},
            {"relative_path": "Dog/dog_0.jpg", "label": "Dog", "split": "train"},
            {"relative_path": "Cat/cat_1.jpg", "label": "Cat", "split": "val"},
            {"relative_path": "Dog/dog_1.jpg", "label": "Dog", "split": "test"},
        ]
        _write_manifest(self.manifest_path, rows)

        self.dataset_cfg = {
            "dataset_root": str(self.dataset_root),
            "manifest_path": str(self.manifest_path),
            "label_map": {"Cat": 0, "Dog": 1},
            "image_size": 224,
            "normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        }
        self.train_cfg = {"batch_size": 2}
        self.runtime_cfg = {"num_parallel_workers": 1, "seed": 7}

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_build_dataloader_returns_expected_batch_shape(self) -> None:
        dataloader = build_dataloader(
            self.dataset_cfg,
            self.train_cfg,
            self.runtime_cfg,
            "train",
        )

        batch = next(dataloader.create_dict_iterator(num_epochs=1, output_numpy=True))
        self.assertEqual(batch["image"].shape, (2, 3, 224, 224))
        self.assertEqual(batch["label"].shape, (2,))
        self.assertEqual(batch["label"].dtype.kind, "i")

    def test_build_dataloaders_returns_all_splits_and_num_classes(self) -> None:
        dataloaders, num_classes = build_dataloaders(
            self.dataset_cfg,
            self.train_cfg,
            self.runtime_cfg,
        )

        self.assertEqual(set(dataloaders), {"train", "val", "test"})
        self.assertEqual(num_classes, 2)

        val_batch = next(dataloaders["val"].create_dict_iterator(num_epochs=1, output_numpy=True))
        test_batch = next(dataloaders["test"].create_dict_iterator(num_epochs=1, output_numpy=True))
        self.assertEqual(val_batch["label"].tolist(), [0])
        self.assertEqual(test_batch["label"].tolist(), [1])
