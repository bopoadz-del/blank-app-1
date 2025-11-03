"""Batch scheduling utilities for RT-DETR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Deque, Iterable, Iterator, List, Optional, Sequence, Tuple

import collections
import logging
import time

import numpy as np

from .config import BatchConfig
from .errors import RTDETRBatchingError

LOGGER = logging.getLogger(__name__)


@dataclass
class Batch:
    items: List[np.ndarray]

    @property
    def size(self) -> int:
        return len(self.items)


@dataclass
class BatchScheduler:
    config: BatchConfig
    queue: Deque[np.ndarray] = field(default_factory=collections.deque)

    def submit(self, item: np.ndarray) -> None:
        LOGGER.debug("Submitting item to batch scheduler queue size=%s", len(self.queue))
        self.queue.append(item)

    def drain(self) -> Iterator[Batch]:
        preferred = self.config.preferred_batch_sizes
        while self.queue:
            batch_size = self._select_batch_size(len(self.queue), preferred)
            items = [self.queue.popleft() for _ in range(batch_size)]
            LOGGER.debug("Yielding batch of size %s", batch_size)
            yield Batch(items)

    def _select_batch_size(self, queue_size: int, preferred: Sequence[int]) -> int:
        for size in sorted(preferred, reverse=True):
            if queue_size >= size:
                return size
        return max(1, min(queue_size, self.config.max_batch_size))

    def warmup(self, iterations: Optional[int] = None) -> None:
        iterations = iterations or self.config.warmup_iterations
        LOGGER.info("Running batch scheduler warmup for %s iterations", iterations)
        for _ in range(iterations):
            time.sleep(0.001)


def collate_batches(batches: Iterable[Batch]) -> List[np.ndarray]:
    return [np.stack(batch.items) for batch in batches if batch.items]


__all__ = ["Batch", "BatchScheduler", "collate_batches"]
