from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ConfigDict = dict[str, Any]


REQUIRED_EXPERIMENT_KEYS = (
    "dataset_config",
    "model_config",
    "train_config",
    "runtime_config",
)


def load_yaml_config(config_path: str | Path) -> ConfigDict:
    """Load a yaml file into a plain dictionary."""
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping at the top level: {path}")
    return data


def resolve_config_path(base_path: str | Path, relative_config_path: str | Path) -> Path:
    """Resolve a nested config path relative to the parent config file."""
    base = Path(base_path).expanduser().resolve()
    relative = Path(relative_config_path).expanduser()
    return relative.resolve() if relative.is_absolute() else (base.parent / relative).resolve()


def infer_project_root(config_path: str | Path) -> Path:
    """Infer repository root from a config file under code/train/configs/."""
    path = Path(config_path).expanduser().resolve()
    for parent in path.parents:
        if (parent / "code" / "train" / "configs").exists():
            return parent
    return path.parent


def load_experiment_config(experiment_config_path: str | Path) -> ConfigDict:
    """Load experiment yaml and inline dataset/model/train/runtime sub-configs."""
    experiment_path = Path(experiment_config_path).expanduser().resolve()
    experiment_cfg = load_yaml_config(experiment_path)

    for key in REQUIRED_EXPERIMENT_KEYS:
        if key not in experiment_cfg:
            raise KeyError(f"Experiment config is missing required key: {key}")

    project_root = infer_project_root(experiment_path)
    merged_config: ConfigDict = {
        "experiment": experiment_cfg,
        "experiment_name": experiment_cfg.get("experiment_name", experiment_path.stem),
        "project_root": str(project_root),
        "config_paths": {"experiment": str(experiment_path)},
    }

    for section_name, experiment_key in (
        ("dataset", "dataset_config"),
        ("model", "model_config"),
        ("train", "train_config"),
        ("runtime", "runtime_config"),
    ):
        section_path = resolve_config_path(experiment_path, experiment_cfg[experiment_key])
        merged_config[section_name] = load_yaml_config(section_path)
        merged_config["config_paths"][section_name] = str(section_path)

    return merged_config
