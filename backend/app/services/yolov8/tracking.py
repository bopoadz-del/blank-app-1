"""Simple multi-object tracker for YOLOv8 detections."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from .config import TrackerConfig
from .postprocessing import Detection

try:  # pragma: no cover
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore


@dataclass
class Track:
    id: int
    bbox: List[float]
    score: float
    class_id: int
    age: int = 0
    hits: int = 0
    missed: int = 0
    embedding: Optional["np.ndarray"] = None

    def update(self, detection: Detection) -> None:
        self.bbox = list(detection.bbox)
        self.score = detection.score
        self.class_id = detection.class_id
        self.age = 0
        self.hits += 1
        self.missed = 0


class MultiObjectTracker:
    def __init__(self, config: TrackerConfig) -> None:
        if np is None:
            raise RuntimeError("NumPy required for tracking")
        self.config = config
        self.next_id = itertools.count(start=1)
        self.tracks: Dict[int, Track] = {}

    def _compute_iou(self, boxes_a: "np.ndarray", boxes_b: "np.ndarray") -> "np.ndarray":
        x11, y11, x12, y12 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
        x21, y21, x22, y22 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]
        xa = np.maximum(x11[:, None], x21[None, :])
        ya = np.maximum(y11[:, None], y21[None, :])
        xb = np.minimum(x12[:, None], x22[None, :])
        yb = np.minimum(y12[:, None], y22[None, :])
        inter = np.maximum(0, xb - xa) * np.maximum(0, yb - ya)
        area_a = (x12 - x11) * (y12 - y11)
        area_b = (x22 - x21) * (y22 - y21)
        return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-6)

    def _associate(self, detections: List[Detection]) -> Dict[int, Detection]:
        if not self.tracks or not detections:
            return {}
        track_ids = list(self.tracks.keys())
        tracks_boxes = np.array([self.tracks[tid].bbox for tid in track_ids], dtype=np.float32)
        det_boxes = np.array([det.bbox for det in detections], dtype=np.float32)
        iou_matrix = self._compute_iou(tracks_boxes, det_boxes)
        matches: Dict[int, Detection] = {}
        for track_idx, track_id in enumerate(track_ids):
            best_det = int(np.argmax(iou_matrix[track_idx]))
            if iou_matrix[track_idx, best_det] >= self.config.iou_threshold:
                matches[track_id] = detections[best_det]
        return matches

    def update(self, detections: Iterable[Detection]) -> List[Track]:
        detections = [det for det in detections if det.score >= self.config.new_track_score_threshold]
        matches = self._associate(detections)

        updated_tracks: List[Track] = []
        assigned = set()
        for track_id, track in list(self.tracks.items()):
            track.age += 1
            if track_id in matches:
                detection = matches[track_id]
                track.update(detection)
                assigned.add(detection)
            else:
                track.missed += 1
            if track.missed > self.config.max_age:
                del self.tracks[track_id]
                continue
            updated_tracks.append(track)

        for detection in detections:
            if detection in assigned:
                continue
            track_id = next(self.next_id)
            track = Track(track_id, list(detection.bbox), detection.score, detection.class_id)
            self.tracks[track_id] = track
            updated_tracks.append(track)

        return [track for track in updated_tracks if track.hits >= self.config.min_hits]


__all__ = ["MultiObjectTracker", "Track"]
