from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def _progress(message: str) -> None:
    print(f"[progress] {message}", flush=True)


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

from data import build_dataloaders
from engine import run_training
from losses import build_loss
from models.shuffleNet import build_model
from utils import (
    build_optimizer,
    configure_context,
    create_run_dirs,
    init_logger,
    load_experiment_config,
    save_config_snapshot,
    set_random_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a MindSpore image classifier.")
    parser.add_argument("--config", required=True, help="Path to experiment yaml config.")
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
        _progress("Known noisy warnings will be suppressed for this run")

    overall_start = time.perf_counter()
    _progress(f"Loading experiment config: {args.config}")
    step_start = time.perf_counter()
    config_bundle = load_experiment_config(args.config)
    _progress(f"Experiment config loaded in {time.perf_counter() - step_start:.2f}s")

    runtime_cfg = config_bundle["runtime"]
    experiment_cfg = config_bundle["experiment"]
    experiment_name = str(experiment_cfg["experiment_name"])

    _progress(f"Configuring MindSpore context for experiment: {experiment_name}")
    step_start = time.perf_counter()
    mode_name = configure_context(runtime_cfg)
    _progress(
        f"MindSpore context configured in {time.perf_counter() - step_start:.2f}s"
        f" ({mode_name})"
    )
    _progress("Setting random seed")
    set_random_seed(int(runtime_cfg.get("seed", 42)))

    _progress("Creating run directories")
    step_start = time.perf_counter()
    run_paths = create_run_dirs(runtime_cfg, experiment_name)
    logger = init_logger(run_paths["log_path"])
    logger.info("Run directory created at: %s", run_paths["run_dir"])
    _progress(f"Run directory ready: {run_paths['run_dir']}")
    _progress(f"Run directories created in {time.perf_counter() - step_start:.2f}s")
    _progress("Saving config snapshot")
    save_config_snapshot(config_bundle, run_paths["config_snapshot_path"])

    _progress("Building dataloaders")
    step_start = time.perf_counter()
    dataloaders, num_classes = build_dataloaders(
        config_bundle["dataset"],
        config_bundle["train"],
        runtime_cfg,
    )
    logger.info(
        "Dataloaders built: train_steps=%s, val_steps=%s, test_steps=%s, num_classes=%s",
        dataloaders["train"].get_dataset_size(),
        dataloaders["val"].get_dataset_size(),
        dataloaders["test"].get_dataset_size(),
        num_classes,
    )
    _progress("Dataloaders built successfully")
    _progress(f"Dataloaders built in {time.perf_counter() - step_start:.2f}s")

    _progress("Building model")
    step_start = time.perf_counter()
    model_cfg = dict(config_bundle["model"])
    model_cfg["num_classes"] = num_classes
    model_cfg["image_size"] = int(config_bundle["dataset"].get("image_size", 224))
    model = build_model(model_cfg)
    _progress(f"Model built in {time.perf_counter() - step_start:.2f}s")
    _progress("Building loss function")
    loss_fn = build_loss(config_bundle["train"])
    _progress("Building optimizer")
    step_start = time.perf_counter()
    optimizer = build_optimizer(model, config_bundle["train"])
    _progress(f"Optimizer built in {time.perf_counter() - step_start:.2f}s")
    _progress("Starting training loop")

    step_start = time.perf_counter()
    result = run_training(
        model=model,
        dataloaders=dataloaders,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_cfg=config_bundle["train"],
        run_paths=run_paths,
        logger=logger,
    )
    logger.info("Training completed. Best checkpoint: %s", result["best_ckpt_path"])
    _progress("Training completed")
    _progress(f"Training loop finished in {time.perf_counter() - step_start:.2f}s")
    _progress(f"Total runtime: {time.perf_counter() - overall_start:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
