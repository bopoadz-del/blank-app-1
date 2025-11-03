"""Postprocessing helpers for YOLOv8 TensorRT inference."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

try:  # pragma: no cover
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

LOGGER = logging.getLogger(__name__)


@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]
    score: float
    class_id: int
    track_id: int = -1


@dataclass
class BatchDetections:
    detections: List[Detection]
    stats: dict


def sigmoid(x: "np.ndarray") -> "np.ndarray":
    return 1.0 / (1.0 + np.exp(-x))


def decode_outputs(outputs: "np.ndarray", anchors: "np.ndarray") -> "np.ndarray":
    if np is None:
        raise RuntimeError("NumPy is required for postprocessing")
    batch, channels, height, width = outputs.shape
    outputs = outputs.reshape(batch, len(anchors), -1, height, width)
    outputs = outputs.transpose(0, 1, 3, 4, 2)
    boxes = outputs[..., :4]
    scores = sigmoid(outputs[..., 4:])
    return boxes, scores


def xywh2xyxy(box: "np.ndarray") -> "np.ndarray":
    x, y, w, h = box.T
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def nms(detections: "np.ndarray", scores: "np.ndarray", iou_threshold: float, max_detections: int) -> List[int]:
    if np is None:
        raise RuntimeError("NumPy is required for postprocessing")
    idxs = scores.argsort()[::-1]
    keep: List[int] = []
    while idxs.size > 0 and len(keep) < max_detections:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        ious = box_iou(detections[i][None, :], detections[idxs[1:]])[0]
        idxs = idxs[1:][ious <= iou_threshold]
    return keep


def box_iou(box1: "np.ndarray", box2: "np.ndarray") -> "np.ndarray":
    if np is None:
        raise RuntimeError("NumPy is required for postprocessing")
    x11, y11, x12, y12 = np.split(box1, 4, axis=-1)
    x21, y21, x22, y22 = np.split(box2, 4, axis=-1)
    xa = np.maximum(x11, x21.T)
    ya = np.maximum(y11, y21.T)
    xb = np.minimum(x12, x22.T)
    yb = np.minimum(y12, y22.T)
    inter_area = np.maximum(0, xb - xa) * np.maximum(0, yb - ya)
    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)
    return inter_area / (area1 + area2.T - inter_area + 1e-6)


def postprocess(
    outputs: "np.ndarray",
    stats: Sequence[dict],
    anchors: "np.ndarray",
    score_threshold: float,
    iou_threshold: float,
    max_detections: int,
    class_names: Sequence[str],
    multi_label: bool = False,
) -> List[BatchDetections]:
    if np is None:
        raise RuntimeError("NumPy is required for postprocessing")

    boxes, scores = decode_outputs(outputs, anchors)
    batch_detections: List[BatchDetections] = []
    for batch_idx, (batch_boxes, batch_scores, info) in enumerate(zip(boxes, scores, stats)):
        candidates: List[Detection] = []
        for anchor_boxes, anchor_scores in zip(batch_boxes.reshape(-1, 4), batch_scores.reshape(-1, len(class_names) + 1)):
            objectness = anchor_scores[0]
            if objectness < score_threshold:
                continue
            class_scores = anchor_scores[1:]
            if multi_label:
                indices = np.where(class_scores > score_threshold)[0]
                for class_id in indices:
                    score = objectness * class_scores[class_id]
                    bbox = xywh2xyxy(anchor_boxes[None, :])[0]
                    candidates.append(Detection(tuple(bbox.tolist()), float(score), int(class_id)))
            else:
                class_id = int(np.argmax(class_scores))
                score = objectness * class_scores[class_id]
                if score < score_threshold:
                    continue
                bbox = xywh2xyxy(anchor_boxes[None, :])[0]
                candidates.append(Detection(tuple(bbox.tolist()), float(score), class_id))

        if not candidates:
            batch_detections.append(BatchDetections([], {"image_index": batch_idx}))
            continue

        boxes_np = np.array([det.bbox for det in candidates], dtype=np.float32)
        scores_np = np.array([det.score for det in candidates], dtype=np.float32)
        keep = nms(boxes_np, scores_np, iou_threshold, max_detections)
        filtered = [candidates[i] for i in keep]
        batch_detections.append(BatchDetections(filtered, {"image_index": batch_idx, **info}))
    return batch_detections


def filter_classes(detections: Iterable[Detection], allowed_classes: Sequence[int]) -> List[Detection]:
    allowed = set(allowed_classes)
    return [det for det in detections if det.class_id in allowed]


__all__ = [
    "Detection",
    "BatchDetections",
    "postprocess",
    "filter_classes",
]
