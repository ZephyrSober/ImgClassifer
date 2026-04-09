from __future__ import annotations

import logging
import random
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import mindspore as ms
import numpy as np
import yaml
from mindspore import context


def _read_config(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)


def configure_context(runtime_cfg: Any) -> str:
    device_target = str(_read_config(runtime_cfg, "device_target", "CPU"))
    device_id = int(_read_config(runtime_cfg, "device_id", 0))
    execution_mode = str(_read_config(runtime_cfg, "execution_mode", "PYNATIVE_MODE")).strip().upper()
    set_context = getattr(ms, "set_context", None) or context.set_context

    if execution_mode == "GRAPH_MODE":
        mode = getattr(ms, "GRAPH_MODE", None)
        if mode is None:
            mode = context.GRAPH_MODE
        mode_name = "GRAPH_MODE"
    elif execution_mode == "PYNATIVE_MODE":
        mode = getattr(ms, "PYNATIVE_MODE", None)
        if mode is None:
            mode = context.PYNATIVE_MODE
        mode_name = "PYNATIVE_MODE"
    else:
        raise ValueError(
            f"Unsupported execution_mode: {execution_mode}. "
            "Expected GRAPH_MODE or PYNATIVE_MODE."
        )

    if device_target.upper() == "CPU":
        set_context(mode=mode, device_target="CPU")
    else:
        set_context(mode=mode, device_target=device_target, device_id=device_id)
    return mode_name


def create_run_dirs(runtime_cfg: Any, experiment_name: str) -> dict[str, Path]:
    run_root = Path(str(_read_config(runtime_cfg, "run_root")))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = run_root / f"{timestamp}_{experiment_name}"
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    metrics_dir = run_dir / "metrics"

    checkpoints_dir.mkdir(parents=True, exist_ok=False)
    logs_dir.mkdir(parents=True, exist_ok=False)
    metrics_dir.mkdir(parents=True, exist_ok=False)

    return {
        "run_dir": run_dir,
        "checkpoints_dir": checkpoints_dir,
        "logs_dir": logs_dir,
        "metrics_dir": metrics_dir,
        "config_snapshot_path": run_dir / "config_snapshot.yaml",
        "log_path": logs_dir / "train.log",
        "epoch_metrics_path": metrics_dir / "epoch_metrics.csv",
        "final_metrics_path": metrics_dir / "final_metrics.json",
    }


def save_config_snapshot(config: Mapping[str, Any], destination: str | Path) -> None:
    with Path(destination).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False, allow_unicode=False)


def init_logger(log_path: str | Path) -> logging.Logger:
    logger = logging.getLogger(f"train_logger_{Path(log_path).stem}_{Path(log_path).parent.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
