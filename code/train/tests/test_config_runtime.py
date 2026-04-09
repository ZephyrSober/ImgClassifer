from __future__ import annotations

import shutil
import sys
import unittest
import uuid
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAIN_SRC = PROJECT_ROOT / "code" / "train" / "src"
TEST_TMP_ROOT = PROJECT_ROOT / "code" / "train" / "tests" / "_tmp"
if str(TRAIN_SRC) not in sys.path:
    sys.path.insert(0, str(TRAIN_SRC))

from utils.config import get_repo_root, load_experiment_config  # noqa: E402
from utils.runtime import create_run_dirs, save_config_snapshot  # noqa: E402


class ConfigRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.root = TEST_TMP_ROOT / f"{self._testMethodName}_{uuid.uuid4().hex}"
        self.root.mkdir(parents=True, exist_ok=True)

        self.relative_dataset_root = "dataset/cleaned_runs/cleaned_petimages_20260317_164602_652096"
        self.relative_manifest_path = (
            "dataset/splits/cleaned_petimages_20260317_164602_652096/"
            "split_test20260317_val20260318/manifest.csv"
        )

        dataset_cfg = self.root / "dataset.yaml"
        dataset_cfg.write_text(
            "\n".join(
                [
                    f"dataset_root: {self.relative_dataset_root}",
                    f"manifest_path: {self.relative_manifest_path}",
                    "label_map:",
                    "  Cat: 0",
                    "  Dog: 1",
                    "image_size: 224",
                    "normalize:",
                    "  mean: [0.485, 0.456, 0.406]",
                    "  std: [0.229, 0.224, 0.225]",
                ]
            ),
            encoding="utf-8",
        )
        (self.root / "model.yaml").write_text(
            "model_name: shufflenet_v2_x1_0\nnum_classes: 2\npretrained: false\n",
            encoding="utf-8",
        )
        (self.root / "train.yaml").write_text(
            "\n".join(
                [
                    "epochs: 1",
                    "batch_size: 2",
                    "optimizer:",
                    "  name: Momentum",
                    "  lr: 0.01",
                    "  momentum: 0.9",
                    "  weight_decay: 0.0001",
                    "loss:",
                    "  name: cross_entropy",
                    "log_interval: 1",
                ]
            ),
            encoding="utf-8",
        )
        (self.root / "runtime.yaml").write_text(
            "device_target: CPU\ndevice_id: 0\nexecution_mode: PYNATIVE_MODE\nseed: 42\nnum_parallel_workers: 1\nrun_root: runs\n",
            encoding="utf-8",
        )
        (self.root / "experiment.yaml").write_text(
            "\n".join(
                [
                    "experiment_name: unit_test_experiment",
                    "dataset_config: ./dataset.yaml",
                    "model_config: ./model.yaml",
                    "train_config: ./train.yaml",
                    "runtime_config: ./runtime.yaml",
                ]
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_load_experiment_config_resolves_relative_paths(self) -> None:
        config_bundle = load_experiment_config(self.root / "experiment.yaml")

        repo_root = get_repo_root()
        self.assertEqual(
            Path(config_bundle["dataset"]["dataset_root"]),
            (repo_root / self.relative_dataset_root).resolve(),
        )
        self.assertEqual(
            Path(config_bundle["dataset"]["manifest_path"]),
            (repo_root / self.relative_manifest_path).resolve(),
        )
        self.assertEqual(
            Path(config_bundle["runtime"]["run_root"]),
            (repo_root / "runs").resolve(),
        )

    def test_create_run_dirs_and_snapshot(self) -> None:
        runtime_cfg = {"run_root": str(self.root / "runs")}
        run_paths = create_run_dirs(runtime_cfg, "demo_experiment")
        self.assertTrue(run_paths["run_dir"].exists())
        self.assertTrue(run_paths["checkpoints_dir"].exists())
        self.assertTrue(run_paths["logs_dir"].exists())
        self.assertTrue(run_paths["metrics_dir"].exists())

        snapshot_payload = {"experiment": {"experiment_name": "demo_experiment"}}
        save_config_snapshot(snapshot_payload, run_paths["config_snapshot_path"])
        self.assertTrue(run_paths["config_snapshot_path"].exists())
