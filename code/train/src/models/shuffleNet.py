from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mindspore import Tensor, nn, ops


_SUPPORTED_MODELS = {"linear_classifier", "shufflenet_v2_x1_0", "simple_cnn"}


def _read_config(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _conv_bn_relu(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    group: int = 1,
) -> nn.SequentialCell:
    return nn.SequentialCell(
        [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                pad_mode="pad",
                padding=padding,
                group=group,
                has_bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
    )


def _conv_bn(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    group: int = 1,
) -> nn.SequentialCell:
    return nn.SequentialCell(
        [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                pad_mode="pad",
                padding=padding,
                group=group,
                has_bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
    )


class ChannelShuffle(nn.Cell):
    def __init__(self, groups: int = 2):
        super().__init__()
        self.groups = groups
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()

    def construct(self, x: Tensor) -> Tensor:
        batch_size, channels, height, width = x.shape
        channels_per_group = channels // self.groups
        x = self.reshape(x, (batch_size, self.groups, channels_per_group, height, width))
        x = self.transpose(x, (0, 2, 1, 3, 4))
        return self.reshape(x, (batch_size, channels, height, width))


class InvertedResidual(nn.Cell):
    def __init__(self, inp: int, outp: int, stride: int):
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(f"stride must be 1 or 2, got {stride}")

        self.stride = stride
        branch_features = outp // 2
        if self.stride == 1 and inp != branch_features * 2:
            raise ValueError("For stride=1, inp must equal outp // 2 * 2.")

        if self.stride > 1:
            self.branch1 = nn.SequentialCell(
                [
                    _conv_bn(inp, inp, kernel_size=3, stride=self.stride, padding=1, group=inp),
                    _conv_bn_relu(inp, branch_features, kernel_size=1, stride=1, padding=0),
                ]
            )
        else:
            self.branch1 = None

        branch2_input = inp if self.stride > 1 else branch_features
        self.branch2 = nn.SequentialCell(
            [
                _conv_bn_relu(branch2_input, branch_features, kernel_size=1, stride=1, padding=0),
                _conv_bn(
                    branch_features,
                    branch_features,
                    kernel_size=3,
                    stride=self.stride,
                    padding=1,
                    group=branch_features,
                ),
                _conv_bn_relu(branch_features, branch_features, kernel_size=1, stride=1, padding=0),
            ]
        )
        self.concat = ops.Concat(axis=1)
        self.split = ops.Split(axis=1, output_num=2)
        self.channel_shuffle = ChannelShuffle(groups=2)

    def construct(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = self.split(x)
            out = self.concat((x1, self.branch2(x2)))
        else:
            out = self.concat((self.branch1(x), self.branch2(x)))
        return self.channel_shuffle(out)


class ShuffleNetV2(nn.Cell):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        stage_repeats = [4, 8, 4]
        stage_out_channels = [24, 116, 232, 464, 1024]

        input_channels = 3
        output_channels = stage_out_channels[0]
        self.conv1 = _conv_bn_relu(input_channels, output_channels, kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")

        input_channels = output_channels
        stages = []
        for repeats, output_channels in zip(stage_repeats, stage_out_channels[1:-1]):
            blocks = [InvertedResidual(input_channels, output_channels, 2)]
            for _ in range(repeats - 1):
                blocks.append(InvertedResidual(output_channels, output_channels, 1))
            stages.append(nn.SequentialCell(blocks))
            input_channels = output_channels

        self.stage2, self.stage3, self.stage4 = stages
        self.conv5 = _conv_bn_relu(input_channels, stage_out_channels[-1], kernel_size=1, stride=1, padding=0)
        self.reduce_mean = ops.ReduceMean(keep_dims=False)
        self.classifier = nn.Dense(stage_out_channels[-1], num_classes)

    def construct(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = self.reduce_mean(x, (2, 3))
        return self.classifier(x)


class SimpleCNN(nn.Cell):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.SequentialCell(
            [
                nn.Conv2d(3, 32, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
        )
        self.reduce_mean = ops.ReduceMean(keep_dims=False)
        self.classifier = nn.SequentialCell(
            [
                nn.Dense(128, 128),
                nn.ReLU(),
                nn.Dense(128, num_classes),
            ]
        )

    def construct(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.reduce_mean(x, (2, 3))
        return self.classifier(x)


class LinearClassifier(nn.Cell):
    def __init__(self, image_size: int = 224, num_classes: int = 2):
        super().__init__()
        flattened_dim = 3 * int(image_size) * int(image_size)
        self.reshape = ops.Reshape()
        self.classifier = nn.Dense(flattened_dim, num_classes)

    def construct(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        x = self.reshape(x, (batch_size, -1))
        return self.classifier(x)


def build_model(config: Any) -> nn.Cell:
    model_name = _read_config(config, "model_name", "shufflenet_v2_x1_0")
    num_classes = int(_read_config(config, "num_classes", 2))
    pretrained = bool(_read_config(config, "pretrained", False))
    image_size = int(_read_config(config, "image_size", 224))

    if model_name not in _SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model_name: {model_name}. Supported models: {sorted(_SUPPORTED_MODELS)}"
        )
    if pretrained:
        raise ValueError("pretrained=True is not supported by the local model implementations.")

    if model_name == "linear_classifier":
        return LinearClassifier(image_size=image_size, num_classes=num_classes)

    if model_name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)

    return ShuffleNetV2(num_classes=num_classes)
