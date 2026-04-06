from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

import numpy as np
from PIL import Image


def _read_config(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, image: Image.Image) -> np.ndarray:
        transformed = image
        for transform in self.transforms:
            transformed = transform(transformed)
        return transformed


class ResizeShortestEdge:
    def __init__(self, size: int):
        self.size = int(size)

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        short_edge = min(width, height)
        if short_edge <= 0:
            raise ValueError(f"Invalid image size: {image.size}")

        scale = self.size / short_edge
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        return image.resize((new_width, new_height), Image.Resampling.BILINEAR)


class RandomCrop:
    def __init__(self, size: int):
        self.size = int(size)

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width < self.size or height < self.size:
            scale = self.size / min(width, height)
            width = max(1, int(round(width * scale)))
            height = max(1, int(round(height * scale)))
            image = image.resize((width, height), Image.Resampling.BILINEAR)

        left = random.randint(0, width - self.size)
        top = random.randint(0, height - self.size)
        return image.crop((left, top, left + self.size, top + self.size))


class CenterCrop:
    def __init__(self, size: int):
        self.size = int(size)

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width < self.size or height < self.size:
            scale = self.size / min(width, height)
            width = max(1, int(round(width * scale)))
            height = max(1, int(round(height * scale)))
            image = image.resize((width, height), Image.Resampling.BILINEAR)

        left = (width - self.size) // 2
        top = (height - self.size) // 2
        return image.crop((left, top, left + self.size, top + self.size))


class RandomHorizontalFlip:
    def __init__(self, probability: float = 0.5):
        self.probability = float(probability)

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.probability:
            return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image


class ToNormalizedTensor:
    def __init__(self, mean: list[float], std: list[float]):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        if self.mean.shape != (3,) or self.std.shape != (3,):
            raise ValueError("normalize.mean and normalize.std must each contain 3 values.")

    def __call__(self, image: Image.Image) -> np.ndarray:
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - self.mean) / self.std
        return np.transpose(array, (2, 0, 1)).astype(np.float32, copy=False)


def build_transforms(dataset_cfg: Any, split: str):
    normalized_split = split.strip().lower()
    if normalized_split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split!r}")

    image_size = int(_read_config(dataset_cfg, "image_size", 224))
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")

    normalize_cfg = _read_config(dataset_cfg, "normalize", {})
    mean = list(_read_config(normalize_cfg, "mean", []))
    std = list(_read_config(normalize_cfg, "std", []))
    resize_short = max(image_size, int(round(image_size / 224 * 256)))

    transforms = [ResizeShortestEdge(resize_short)]
    if normalized_split == "train":
        transforms.extend([RandomCrop(image_size), RandomHorizontalFlip(0.5)])
    else:
        transforms.append(CenterCrop(image_size))
    transforms.append(ToNormalizedTensor(mean, std))
    return Compose(transforms)
