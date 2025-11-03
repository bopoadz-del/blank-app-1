"""Postprocessing routines for RT-DETR detections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import logging
import numpy as np

from .config import PostprocessingConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class Detection:
    class_id: int
    score: float
    box: Tuple[float, float, float, float]


@dataclass
class PostprocessingResult:
    detections: List[Detection]
    raw_logits: np.ndarray
    raw_boxes: np.ndarray


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float, topk: int) -> List[int]:
    LOGGER.debug("Running NMS on %s boxes", len(boxes))
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1][:topk]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]
    return keep


def cxcywh_to_xyxy(box: Sequence[float]) -> Tuple[float, float, float, float]:
    cx, cy, w, h = box
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def scale_boxes(boxes: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
    LOGGER.debug("Scaling boxes to original shape %s", original_shape)
    height, width = original_shape
    scale = np.array([width, height, width, height], dtype=np.float32)
    return boxes * scale


def postprocess(
    logits: np.ndarray,
    boxes: np.ndarray,
    original_shape: Tuple[int, int],
    config: PostprocessingConfig,
) -> PostprocessingResult:
    LOGGER.debug(
        "Postprocessing logits shape=%s boxes shape=%s", logits.shape, boxes.shape
    )
    class_probs = softmax(logits, axis=-1)
    scores = class_probs.max(axis=-1)
    class_ids = class_probs.argmax(axis=-1)
    threshold_mask = scores >= config.conf_threshold
    filtered_scores = scores[threshold_mask]
    filtered_class_ids = class_ids[threshold_mask]
    filtered_boxes = boxes[threshold_mask]
    if config.box_format == "cxcywh":
        filtered_boxes = np.array([cxcywh_to_xyxy(box) for box in filtered_boxes])
    scaled_boxes = scale_boxes(filtered_boxes, original_shape)
    keep_indices = nms(scaled_boxes, filtered_scores, config.nms_threshold, config.topk)
    detections = [
        Detection(
            class_id=int(filtered_class_ids[idx]),
            score=float(filtered_scores[idx]),
            box=tuple(map(float, scaled_boxes[idx])),
        )
        for idx in keep_indices[: config.max_detections]
    ]
    return PostprocessingResult(detections=detections, raw_logits=logits, raw_boxes=boxes)


def batch_postprocess(
    logits_batch: np.ndarray,
    boxes_batch: np.ndarray,
    original_shapes: Iterable[Tuple[int, int]],
    config: PostprocessingConfig,
) -> List[PostprocessingResult]:
    results = []
    for logits, boxes, shape in zip(logits_batch, boxes_batch, original_shapes):
        results.append(postprocess(logits, boxes, shape, config))
    return results


__all__ = [
    "Detection",
    "PostprocessingResult",
    "postprocess",
    "batch_postprocess",
    "sigmoid",
    "softmax",
    "nms",
    "cxcywh_to_xyxy",
    "scale_boxes",
]
