"""Inference services for RT-DETR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

import logging
import threading
import time

import numpy as np

from .config import RTDETRConfig
from .errors import RTDETRTimeoutError
from .postprocessing import PostprocessingConfig, batch_postprocess
from .preprocessing import ImageBatch, PreprocessingPipeline
from .tensorrt_builder import EngineBuildResult

LOGGER = logging.getLogger(__name__)


@dataclass
class RTDETRInferenceSession:
    config: RTDETRConfig
    engine: EngineBuildResult
    preprocessing: PreprocessingPipeline
    postprocessing_config: PostprocessingConfig

    def infer(self, images: Sequence[np.ndarray], *, timeout_s: Optional[float] = None) -> List[dict]:
        LOGGER.debug("Starting inference for batch of %s images", len(images))
        start = time.perf_counter()
        batch = self.preprocessing.process(images)
        outputs = self._simulate_engine(batch.arrays)
        detections = batch_postprocess(
            outputs["logits"],
            outputs["boxes"],
            batch.original_shapes,
            self.postprocessing_config,
        )
        duration = time.perf_counter() - start
        if timeout_s and duration > timeout_s:
            raise RTDETRTimeoutError(
                f"Inference exceeded timeout ({duration:.2f}s > {timeout_s}s)",
                hint="Reduce batch size or increase timeout",
            )
        LOGGER.info("Inference completed in %.3fs", duration)
        return [
            {
                "detections": [det.__dict__ for det in result.detections],
                "raw_logits_shape": result.raw_logits.shape,
                "raw_boxes_shape": result.raw_boxes.shape,
            }
            for result in detections
        ]

    def _simulate_engine(self, inputs: np.ndarray) -> dict:
        LOGGER.debug("Simulating TensorRT engine execution with inputs shape %s", inputs.shape)
        batch_size, _, height, width = inputs.shape
        logits = np.random.rand(batch_size, self.config.model.num_queries, self.config.model.num_classes)
        boxes = np.random.rand(batch_size, self.config.model.num_queries, 4)
        return {"logits": logits, "boxes": boxes}


class RTDETRService:
    """High-level API for RT-DETR inference."""

    def __init__(self, session: RTDETRInferenceSession) -> None:
        self.session = session

    def predict(self, images: Sequence[np.ndarray], timeout_s: Optional[float] = None) -> List[dict]:
        LOGGER.debug("RTDETRService.predict invoked")
        return self.session.infer(images, timeout_s=timeout_s)


class HotReloadWatcher(threading.Thread):
    """Watches engine files and triggers reloads when they change."""

    def __init__(self, session_factory, engine_path: Path, interval_s: float) -> None:
        super().__init__(daemon=True)
        self.session_factory = session_factory
        self.engine_path = engine_path
        self.interval_s = interval_s
        self._stop_event = threading.Event()
        self._last_mtime: Optional[float] = None

    def run(self) -> None:
        LOGGER.info("Starting hot reload watcher for %s", self.engine_path)
        while not self._stop_event.is_set():
            try:
                mtime = self.engine_path.stat().st_mtime
            except FileNotFoundError:
                LOGGER.debug("Engine path %s not found", self.engine_path)
                time.sleep(self.interval_s)
                continue
            if self._last_mtime is None:
                self._last_mtime = mtime
            elif mtime != self._last_mtime:
                LOGGER.info("Engine change detected; rebuilding session")
                self.session_factory()
                self._last_mtime = mtime
            time.sleep(self.interval_s)

    def stop(self) -> None:
        self._stop_event.set()


def create_inference_session(config: RTDETRConfig, engine: EngineBuildResult) -> RTDETRInferenceSession:
    preprocessing_config = config.preprocessing
    preprocessing_config.dataset_shape = config.dataset.image_size
    preprocessing = PreprocessingPipeline(preprocessing_config)
    session = RTDETRInferenceSession(
        config=config,
        engine=engine,
        preprocessing=preprocessing,
        postprocessing_config=config.postprocessing,
    )
    return session
