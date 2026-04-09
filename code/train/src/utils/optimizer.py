from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mindspore import nn


_SUPPORTED_OPTIMIZERS = {"ADAM", "MOMENTUM", "SGD"}


def _read_config(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def build_optimizer(model, train_cfg: Any):
    optimizer_cfg = _read_config(train_cfg, "optimizer", {})
    optimizer_name = str(_read_config(optimizer_cfg, "name", "MOMENTUM")).strip().upper()
    if optimizer_name not in _SUPPORTED_OPTIMIZERS:
        raise ValueError(
            f"Unsupported optimizer.name: {optimizer_name}. "
            f"Supported optimizers: {sorted(_SUPPORTED_OPTIMIZERS)}"
        )

    lr = float(_read_config(optimizer_cfg, "lr", 0.01))
    momentum = float(_read_config(optimizer_cfg, "momentum", 0.9))
    weight_decay = float(_read_config(optimizer_cfg, "weight_decay", 0.0))

    if optimizer_name == "ADAM":
        return nn.Adam(
            params=model.trainable_params(),
            learning_rate=lr,
            weight_decay=weight_decay,
        )

    if optimizer_name == "MOMENTUM":
        return nn.Momentum(
            params=model.trainable_params(),
            learning_rate=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    return nn.SGD(
        params=model.trainable_params(),
        learning_rate=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
