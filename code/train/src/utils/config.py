from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]


def _read_config(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must contain a mapping: {config_path}")
    return data


def _resolve_repo_path(value: str | Path) -> Path:
    return (REPO_ROOT / Path(value)).resolve()


def load_experiment_config(experiment_path: str | Path) -> dict[str, Any]:
    experiment_path = Path(experiment_path).resolve()
    experiment_cfg = load_yaml(experiment_path)
    experiment_dir = experiment_path.parent

    dataset_cfg = load_yaml(experiment_dir / experiment_cfg["dataset_config"])
    model_cfg = load_yaml(experiment_dir / experiment_cfg["model_config"])
    train_cfg = load_yaml(experiment_dir / experiment_cfg["train_config"])
    runtime_cfg = load_yaml(experiment_dir / experiment_cfg["runtime_config"])

    dataset_cfg["dataset_root"] = str(_resolve_repo_path(dataset_cfg["dataset_root"]))
    dataset_cfg["manifest_path"] = str(_resolve_repo_path(dataset_cfg["manifest_path"]))
    runtime_cfg["run_root"] = str(_resolve_repo_path(runtime_cfg["run_root"]))

    return {
        "experiment_path": str(experiment_path),
        "experiment": experiment_cfg,
        "dataset": dataset_cfg,
        "model": model_cfg,
        "train": train_cfg,
        "runtime": runtime_cfg,
    }


def get_repo_root() -> Path:
    return REPO_ROOT
