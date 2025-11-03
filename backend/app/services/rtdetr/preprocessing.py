"""High-throughput preprocessing pipeline for RT-DETR."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import logging
import math
import numpy as np

from .config import PreprocessingConfig
from .errors import RTDETRConfigurationError

LOGGER = logging.getLogger(__name__)


@dataclass
class ImageBatch:
    paths: List[Path]
    arrays: np.ndarray
    original_shapes: List[Tuple[int, int]]


def letterbox(image: np.ndarray, target_shape: Tuple[int, int], pad_value: float) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
    LOGGER.debug("Applying letterbox to image of shape %s", image.shape)
    height, width = image.shape[:2]
    target_height, target_width = target_shape
    scale = min(target_height / height, target_width / width)
    resized_height = int(round(height * scale))
    resized_width = int(round(width * scale))
    resized = np.zeros((target_height, target_width, 3), dtype=image.dtype)
    pad_top = (target_height - resized_height) // 2
    pad_left = (target_width - resized_width) // 2
    resized[pad_top : pad_top + resized_height, pad_left : pad_left + resized_width] = np.resize(
        image, (resized_height, resized_width, 3)
    )
    return resized, (scale, scale), (pad_left, pad_top)


def normalize(image: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    LOGGER.debug("Normalizing image with mean=%s std=%s", mean, std)
    return (image / 255.0 - mean) / std


def to_chw(image: np.ndarray) -> np.ndarray:
    return np.transpose(image, (2, 0, 1))


@dataclass
class PreprocessingPipeline:
    config: PreprocessingConfig

    def _allocate(self, batch_size: int, shape: Tuple[int, int]) -> np.ndarray:
        LOGGER.debug("Allocating preprocessing buffer for batch=%s shape=%s", batch_size, shape)
        channels = 3
        height, width = shape
        return np.empty((batch_size, channels, height, width), dtype=np.float32)

    def process(self, images: Sequence[np.ndarray], *, image_paths: Optional[Sequence[Path]] = None) -> ImageBatch:
        if not images:
            raise RTDETRConfigurationError("No images provided to preprocessing pipeline")
        if image_paths and len(image_paths) != len(images):
            raise RTDETRConfigurationError("image_paths must match the number of images")
        target_shape = tuple(self.config.dataset_shape)
        batch_size = len(images)
        buffer = self._allocate(batch_size, target_shape)
        original_shapes: List[Tuple[int, int]] = []
        for index, image in enumerate(images):
            original_shapes.append(image.shape[:2])
            processed = image.astype(np.float32)
            if self.config.letterbox:
                processed, _, _ = letterbox(processed, target_shape, self.config.pad_value)
            if self.config.normalize:
                processed = normalize(processed, self.config.mean, self.config.std)
            buffer[index] = to_chw(processed)
        paths = [Path(p) for p in image_paths] if image_paths else [Path(f"image_{i}.jpg") for i in range(batch_size)]
        return ImageBatch(paths=paths, arrays=buffer, original_shapes=original_shapes)


def iterate_batches(images: Iterable[np.ndarray], batch_size: int) -> Iterator[List[np.ndarray]]:
    batch: List[np.ndarray] = []
    for image in images:
        batch.append(image)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def compute_stride(shape: Tuple[int, int], stride: int) -> Tuple[int, int]:
    height, width = shape
    return (int(math.ceil(height / stride) * stride), int(math.ceil(width / stride) * stride))


__all__ = [
    "ImageBatch",
    "PreprocessingPipeline",
    "letterbox",
    "normalize",
    "to_chw",
    "iterate_batches",
    "compute_stride",
]
