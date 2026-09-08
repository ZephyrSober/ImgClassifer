from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


class _FilteredStream:
    _SUPPRESSED_PATTERNS = (
        "WARNING: 'ControlDepend' is deprecated",
        "WARN_DEPRECATED: The usage of Pack is deprecated. Please use Stack.",
    )

    def __init__(self, wrapped):
        self._wrapped = wrapped
        self._buffer = ""

    def write(self, text):
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if not any(pattern in line for pattern in self._SUPPRESSED_PATTERNS):
                self._wrapped.write(line + "\n")
        return len(text)

    def flush(self):
        if self._buffer and not any(
            pattern in self._buffer for pattern in self._SUPPRESSED_PATTERNS
        ):
            self._wrapped.write(self._buffer)
        self._buffer = ""
        self._wrapped.flush()

    def isatty(self):
        return self._wrapped.isatty()


def _enable_known_warning_filter() -> None:
    sys.stderr = _FilteredStream(sys.stderr)
    sys.stdout = _FilteredStream(sys.stdout)


def _enable_known_warning_env() -> None:
    os.environ["GLOG_v"] = "3"
    os.environ["GLOG_stderrthreshold"] = "3"
    os.environ["MS_SUBMODULE_LOG_v"] = "3"
    os.environ["PYTHONWARNINGS"] = "ignore"


_EARLY_SUPPRESS = "--suppress-known-warnings" in sys.argv
if _EARLY_SUPPRESS:
    _enable_known_warning_env()
    _enable_known_warning_filter()


TRAIN_SRC = Path(__file__).resolve().parents[1] / "src"
if str(TRAIN_SRC) not in sys.path:
    sys.path.insert(0, str(TRAIN_SRC))

import mindspore as ms

from data import build_dataloader
from engine import validate_one_epoch
from losses import build_loss
from models.shuffleNet import build_model
from utils import configure_context, load_experiment_config, set_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a MindSpore image classifier.")
    parser.add_argument("--config", required=True, help="Path to experiment yaml config.")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint file.")
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="Dataset split to evaluate. Defaults to test.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write JSON metrics.",
    )
    parser.add_argument(
        "--suppress-known-warnings",
        action="store_true",
        help="Suppress known noisy MindSpore deprecation warnings in this run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.suppress_known_warnings and not _EARLY_SUPPRESS:
        _enable_known_warning_env()
        _enable_known_warning_filter()

    config_bundle = load_experiment_config(args.config)
    runtime_cfg = config_bundle["runtime"]
    configure_context(runtime_cfg)
    set_random_seed(int(runtime_cfg.get("seed", 42)))

    label_map = config_bundle["dataset"].get("label_map", {})
    model_cfg = dict(config_bundle["model"])
    model_cfg["num_classes"] = len(label_map)
    model_cfg["image_size"] = int(config_bundle["dataset"].get("image_size", 224))

    dataloader = build_dataloader(
        config_bundle["dataset"],
        config_bundle["train"],
        runtime_cfg,
        args.split,
    )
    model = build_model(model_cfg)
    loss_fn = build_loss(config_bundle["train"])

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")
    params = ms.load_checkpoint(str(ckpt_path))
    ms.load_param_into_net(model, params)

    loss, accuracy = validate_one_epoch(model, dataloader, loss_fn)
    result = {
        "config": str(Path(args.config).resolve()),
        "ckpt": str(ckpt_path),
        "split": args.split,
        "loss": loss,
        "accuracy": accuracy,
        "steps": dataloader.get_dataset_size(),
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
