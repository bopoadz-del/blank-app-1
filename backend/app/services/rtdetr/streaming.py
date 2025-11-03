"""Multi-stream orchestration for RT-DETR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

import logging
import queue
import threading
import time

import numpy as np

from .config import MultiStreamConfig
from .errors import RTDETRStreamError
from .batching import BatchScheduler

LOGGER = logging.getLogger(__name__)


@dataclass
class StreamResult:
    stream_id: int
    batch_index: int
    output: List[dict]


class StreamWorker(threading.Thread):
    def __init__(
        self,
        stream_id: int,
        task_queue: "queue.Queue[tuple[int, np.ndarray]]",
        result_queue: "queue.Queue[StreamResult]",
        process_fn: Callable[[np.ndarray], List[dict]],
        shutdown_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self.stream_id = stream_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.process_fn = process_fn
        self.shutdown_event = shutdown_event

    def run(self) -> None:
        LOGGER.info("Stream worker %s starting", self.stream_id)
        while not self.shutdown_event.is_set():
            try:
                batch_index, batch = self.task_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                output = self.process_fn(batch)
                self.result_queue.put(StreamResult(self.stream_id, batch_index, output))
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.exception("Stream %s failed: %s", self.stream_id, exc)
                raise RTDETRStreamError(f"Stream {self.stream_id} processing failed", hint=str(exc)) from exc
            finally:
                self.task_queue.task_done()
        LOGGER.info("Stream worker %s stopping", self.stream_id)


@dataclass
class MultiStreamOrchestrator:
    config: MultiStreamConfig
    scheduler: BatchScheduler
    process_fn: Callable[[np.ndarray], List[dict]]
    _task_queue: "queue.Queue[tuple[int, np.ndarray]]" = field(
        default_factory=lambda: queue.Queue(maxsize=32)
    )
    _result_queue: "queue.Queue[StreamResult]" = field(
        default_factory=lambda: queue.Queue(maxsize=32)
    )
    _shutdown: threading.Event = field(default_factory=threading.Event)
    _workers: List[StreamWorker] = field(default_factory=list)

    def start(self) -> None:
        LOGGER.info("Starting multi-stream orchestrator with %s streams", self.config.stream_count)
        if self._workers:
            LOGGER.debug("Multi-stream orchestrator already running")
            return
        for stream_id in range(self.config.stream_count):
            worker = StreamWorker(
                stream_id=stream_id,
                task_queue=self._task_queue,
                result_queue=self._result_queue,
                process_fn=self.process_fn,
                shutdown_event=self._shutdown,
            )
            worker.start()
            self._workers.append(worker)

    def stop(self) -> None:
        LOGGER.info("Stopping multi-stream orchestrator")
        self._shutdown.set()
        for worker in self._workers:
            worker.join(timeout=1.0)
        self._workers.clear()

    def submit_batches(self, batches: Iterable[np.ndarray]) -> None:
        for index, batch in enumerate(batches):
            LOGGER.debug("Submitting batch %s to stream queue", index)
            self._task_queue.put((index, batch))

    def collect_results(self, expected: int, timeout_s: float = 10.0) -> List[StreamResult]:
        results: List[StreamResult] = []
        start = time.perf_counter()
        while len(results) < expected:
            remaining = timeout_s - (time.perf_counter() - start)
            if remaining <= 0:
                raise RTDETRStreamError("Timed out waiting for stream results")
            try:
                result = self._result_queue.get(timeout=remaining)
                results.append(result)
            except queue.Empty:
                continue
        return results


__all__ = ["MultiStreamOrchestrator", "StreamResult"]
