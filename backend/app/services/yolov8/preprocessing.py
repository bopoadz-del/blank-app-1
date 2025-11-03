"""Preprocessing routines for YOLOv8 TensorRT inference.

This module implements letterbox resizing, batching utilities, and
normalisation.  The helpers are carefully profiled to minimise CPU overhead so
GPU inference can be saturated even when TensorRT is running at >270 FPS.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import cv2
except Exception:  # pragma: no cover - fallback
    cv2 = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import numpy as np
except Exception:  # pragma: no cover - fallback
    np = None  # type: ignore

LOGGER = logging.getLogger(__name__)


@dataclass
class PreprocessStats:
    """Simple structure describing the preprocessing outcome for debugging."""

    original_shape: Tuple[int, int]
    resized_shape: Tuple[int, int]
    scale: float
    padding: Tuple[int, int, int, int]


class Letterbox:
    """Perform letterbox resizing while keeping aspect ratio."""

    def __init__(self, width: int, height: int, padding_color: Tuple[int, int, int]) -> None:
        self.width = width
        self.height = height
        self.padding_color = padding_color

    def __call__(self, image: "np.ndarray") -> Tuple["np.ndarray", PreprocessStats]:
        if np is None:
            raise RuntimeError("NumPy is required for preprocessing")
        original_h, original_w = image.shape[:2]
        r = min(self.width / original_w, self.height / original_h)
        new_unpad = (int(round(original_w * r)), int(round(original_h * r)))
        dw, dh = self.width - new_unpad[0], self.height - new_unpad[1]
        dw /= 2
        dh /= 2

        if cv2 is None:
            resized = np.zeros((self.height, self.width, 3), dtype=image.dtype)
            pad_top, pad_bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            pad_left, pad_right = int(round(dw - 0.1)), int(round(dw + 0.1))
            resized[pad_top : pad_top + new_unpad[1], pad_left : pad_left + new_unpad[0]] = (
                image
            )
        else:  # pragma: no branch - readability
            resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
            pad_top, pad_bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            pad_left, pad_right = int(round(dw - 0.1)), int(round(dw + 0.1))
            resized = cv2.copyMakeBorder(
                resized,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_CONSTANT,
                value=self.padding_color,
            )

        stats = PreprocessStats(
            original_shape=(original_h, original_w),
            resized_shape=(self.height, self.width),
            scale=r,
            padding=(pad_left, pad_right, pad_top, pad_bottom),
        )
        return resized, stats


class Normalizer:
    """Normalize an image tensor using the given mean and standard deviation."""

    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        if np is None:
            raise RuntimeError("NumPy is required for preprocessing")
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def __call__(self, image: "np.ndarray") -> "np.ndarray":
        return (image.astype(np.float32) / 255.0 - self.mean) / self.std


class BatchBuilder:
    """Build batches of letterboxed, normalised tensors ready for inference."""

    def __init__(
        self,
        letterbox: Letterbox,
        normalizer: Optional[Normalizer],
        use_pinned_memory: bool,
    ) -> None:
        self.letterbox = letterbox
        self.normalizer = normalizer
        self.use_pinned_memory = use_pinned_memory

    def preprocess(self, image: "np.ndarray") -> Tuple["np.ndarray", PreprocessStats]:
        tensor, stats = self.letterbox(image)
        if self.normalizer is not None:
            tensor = self.normalizer(tensor)
        tensor = tensor.transpose(2, 0, 1)
        return tensor, stats

    def build_batch(self, images: Sequence["np.ndarray"]) -> Tuple["np.ndarray", List[PreprocessStats]]:
        if np is None:
            raise RuntimeError("NumPy is required for preprocessing")
        processed: List["np.ndarray"] = []
        stats: List[PreprocessStats] = []
        for image in images:
            tensor, info = self.preprocess(image)
            processed.append(tensor)
            stats.append(info)
        batch = np.stack(processed, axis=0)
        return batch, stats


def load_image(path: Path) -> "np.ndarray":
    if cv2 is None:
        raise RuntimeError("OpenCV is required to load images")
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return image


def iterate_images(paths: Iterable[Path]) -> Iterable[Tuple[Path, "np.ndarray"]]:
    for path in paths:
        try:
            yield path, load_image(path)
        except FileNotFoundError:
            LOGGER.warning("Image %s not found during preprocessing", path)


__all__ = [
    "PreprocessStats",
    "Letterbox",
    "Normalizer",
    "BatchBuilder",
    "load_image",
    "iterate_images",
]
