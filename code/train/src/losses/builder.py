from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mindspore import nn


def _read_config(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def build_loss(train_cfg: Any):
    loss_cfg = _read_config(train_cfg, "loss", {})
    loss_name = str(_read_config(loss_cfg, "name", "cross_entropy")).strip().lower()
    if loss_name != "cross_entropy":
        raise ValueError(f"Unsupported loss.name: {loss_name}")
    return nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
