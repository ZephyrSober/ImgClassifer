from __future__ import annotations

import shutil
import sys
import unittest
import uuid
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAIN_SRC = PROJECT_ROOT / "code" / "train" / "src"
TEST_TMP_ROOT = PROJECT_ROOT / "code" / "train" / "tests" / "_tmp"
if str(TRAIN_SRC) not in sys.path:
    sys.path.insert(0, str(TRAIN_SRC))

try:
    from mindspore import nn
    import mindspore.dataset as ds
except ImportError:
    MINDSPORE_AVAILABLE = False
else:
    MINDSPORE_AVAILABLE = True

from engine import save_checkpoint_if_best, validate_one_epoch  # noqa: E402
from losses import build_loss  # noqa: E402
from models.shuffleNet import build_model  # noqa: E402
from utils.optimizer import build_optimizer  # noqa: E402


@unittest.skipUnless(MINDSPORE_AVAILABLE, "mindspore is required for builder and engine tests")
class BuilderAndEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.root = TEST_TMP_ROOT / f"{self._testMethodName}_{uuid.uuid4().hex}"
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_build_loss_returns_cross_entropy(self) -> None:
        loss_fn = build_loss({"loss": {"name": "cross_entropy"}})
        self.assertIsInstance(loss_fn, nn.SoftmaxCrossEntropyWithLogits)

    def test_build_loss_rejects_unknown_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported loss.name"):
            build_loss({"loss": {"name": "mse"}})

    def test_build_optimizer_returns_momentum(self) -> None:
        model = nn.Dense(4, 2)
        optimizer = build_optimizer(
            model,
            {
                "optimizer": {
                    "name": "Momentum",
                    "lr": 0.01,
                    "momentum": 0.9,
                    "weight_decay": 0.0001,
                }
            },
        )
        self.assertIsInstance(optimizer, nn.Momentum)

    def test_build_optimizer_supports_sgd(self) -> None:
        model = nn.Dense(4, 2)
        optimizer = build_optimizer(
            model,
            {
                "optimizer": {
                    "name": "SGD",
                    "lr": 0.01,
                    "momentum": 0.9,
                    "weight_decay": 0.0001,
                }
            },
        )
        self.assertIsInstance(optimizer, nn.SGD)

    def test_build_optimizer_rejects_unknown_name(self) -> None:
        model = nn.Dense(4, 2)
        with self.assertRaisesRegex(ValueError, "Unsupported optimizer.name"):
            build_optimizer(model, {"optimizer": {"name": "RMSProp"}})

    def test_build_optimizer_supports_adam(self) -> None:
        model = nn.Dense(4, 2)
        optimizer = build_optimizer(
            model,
            {
                "optimizer": {
                    "name": "Adam",
                    "lr": 0.0001,
                    "weight_decay": 0.0,
                }
            },
        )
        self.assertIsInstance(optimizer, nn.Adam)

    def test_build_model_supports_simple_cnn(self) -> None:
        model = build_model(
            {
                "model_name": "simple_cnn",
                "num_classes": 2,
                "pretrained": False,
            }
        )
        self.assertIsInstance(model, nn.Cell)

    def test_build_model_supports_linear_classifier(self) -> None:
        model = build_model(
            {
                "model_name": "linear_classifier",
                "num_classes": 2,
                "pretrained": False,
                "image_size": 224,
            }
        )
        self.assertIsInstance(model, nn.Cell)

    def test_save_checkpoint_if_best_only_updates_when_improved(self) -> None:
        model = nn.Dense(4, 2)
        best_acc, best_path = save_checkpoint_if_best(
            model,
            current_acc=0.5,
            best_acc=-1.0,
            save_dir=self.root,
            model_name="best.ckpt",
            epoch=1,
        )
        self.assertEqual(best_acc, 0.5)
        self.assertIsNotNone(best_path)
        self.assertTrue((self.root / "best_epoch_001.ckpt").exists())

        best_acc_second, best_path_second = save_checkpoint_if_best(
            model,
            current_acc=0.4,
            best_acc=best_acc,
            save_dir=self.root,
            model_name="best.ckpt",
            epoch=2,
        )
        self.assertEqual(best_acc_second, 0.5)
        self.assertIsNone(best_path_second)

    def test_validate_one_epoch_returns_loss_and_accuracy(self) -> None:
        class FixedModel(nn.Cell):
            def construct(self, x):
                return x

        logits = np.asarray([[5.0, 0.1], [0.1, 4.0]], dtype=np.float32)
        labels = np.asarray([0, 1], dtype=np.int32)
        dataset = ds.NumpySlicesDataset(
            {"image": logits, "label": labels},
            shuffle=False,
        ).batch(2)

        model = FixedModel()
        loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        loss, accuracy = validate_one_epoch(model, dataset, loss_fn)

        self.assertGreaterEqual(loss, 0.0)
        self.assertAlmostEqual(accuracy, 1.0, places=6)
