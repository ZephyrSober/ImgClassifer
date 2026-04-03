from collections.abc import Mapping
from typing import Any

from mindspore import nn
from mindcv.models import shufflenet_v2_x1_0


_SUPPORTED_MODEL = "shufflenet_v2_x1_0"


def _read_config(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def build_model(config: Any) -> nn.Cell:
    """Build model from config.

    Required config fields:
    - model_name
    - num_classes
    - pretrained
    """
    model_name = _read_config(config, "model_name", _SUPPORTED_MODEL)
    num_classes = int(_read_config(config, "num_classes", 2))
    pretrained = bool(_read_config(config, "pretrained", False))

    if model_name != _SUPPORTED_MODEL:
        raise ValueError(f"Unsupported model_name: {model_name}")

    return shufflenet_v2_x1_0(pretrained=pretrained, num_classes=num_classes)
