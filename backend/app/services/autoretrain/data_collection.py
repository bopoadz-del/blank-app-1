<<<<< codex/debug-application-issues-vqhnb4
"""Utilities for collecting raw samples from configurable sources."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from random import Random
from typing import List

from .config import DataCollectionConfig
from .entities import RawSample


class DataCollector:
    """Aggregate data from multiple configured sources."""

    def __init__(self, config: DataCollectionConfig, *, seed: int = 13) -> None:
        self._config = config
        self._random = Random(seed)

    def collect_batch(self) -> List[RawSample]:
        """Collect and normalise samples from all configured sources."""

        samples: List[RawSample] = []
        for name, source in self._config.sources.items():
            samples.extend(self._coerce_samples(name, source))

        if self._config.shuffle:
            self._random.shuffle(samples)

        if self._config.limit is not None:
            samples = samples[: self._config.limit]

        return samples

    def _coerce_samples(self, source_name: str, data: Iterable[Mapping[str, object]]) -> List[RawSample]:
        samples: List[RawSample] = []
        for idx, record in enumerate(data):
            try:
                features = [float(value) for value in self._as_sequence(record.get("features"))]
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive programming
                raise ValueError(f"Source {source_name} produced invalid features at index {idx}") from exc

            label_obj = record.get("label")
            label = None if label_obj is None else int(label_obj)
            metadata = {
                key: float(value)
                for key, value in record.items()
                if key not in {"features", "label"} and isinstance(value, (int, float))
            }
            samples.append(RawSample(features=features, label=label, metadata=metadata))
        return samples

    @staticmethod
    def _as_sequence(value: object) -> Iterable[float]:
        if value is None:
            raise ValueError("features cannot be None")
        if isinstance(value, (list, tuple)):
            return value
        raise TypeError("features must be a sequence of numeric values")
=======
"""Data collection services for automated retraining."""
from __future__ import annotations

import json
import random
import threading
import time
from collections import deque
from datetime import datetime
from typing import Deque, Dict, Iterable, Iterator, List, Optional

from .config import DataCollectionConfig


class DataCollector:
    """Collects and buffers data for the automated retraining pipeline."""

    def __init__(self, config: DataCollectionConfig) -> None:
        config.validate()
        self._config = config
        self._buffer: Deque[Dict[str, object]] = deque(maxlen=config.batch_size * 10)
        self._lock = threading.Lock()
        self._last_flush = datetime.utcnow()
        self._cache_dir = config.cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def config(self) -> DataCollectionConfig:
        return self._config

    def ingest(self, record: Dict[str, object]) -> None:
        """Add a single record to the buffer."""

        with self._lock:
            self._buffer.append(record)
            if len(self._buffer) >= self._config.batch_size:
                self._flush_locked()

    def collect_from_source(self, source_id: str) -> Iterator[Dict[str, object]]:
        """Yield records from a configured data source."""

        uri = self._config.sources.get(source_id)
        if not uri:
            raise KeyError(f"Unknown data source: {source_id}")
        random.seed(hash(uri))
        for idx in range(self._config.batch_size):
            yield {
                "source": source_id,
                "timestamp": datetime.utcnow().isoformat(),
                "payload": {"feature": random.random(), "index": idx},
            }

    def collect_batch(self) -> List[Dict[str, object]]:
        """Collect a batch from all configured sources."""

        batch: List[Dict[str, object]] = []
        for source_id in self._config.sources:
            batch.extend(self.collect_from_source(source_id))
        for record in batch:
            self.ingest(record)
        return batch

    def drain(self) -> List[Dict[str, object]]:
        """Retrieve and clear the buffer."""

        with self._lock:
            drained = list(self._buffer)
            self._buffer.clear()
            return drained

    def _flush_locked(self) -> None:
        """Persist the buffer to disk."""

        now = datetime.utcnow()
        if now - self._last_flush < self._config.poll_interval:
            return
        payload = list(self._buffer)
        if not payload:
            return
        path = self._cache_dir / f"batch-{int(now.timestamp())}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        self._buffer.clear()
        self._last_flush = now

    def auto_flush(self, stop_event: Optional[threading.Event] = None) -> None:
        """Periodically flush the buffer to disk."""

        stop_event = stop_event or threading.Event()
        while not stop_event.is_set():
            time.sleep(self._config.poll_interval.total_seconds())
            with self._lock:
                self._flush_locked()

    def replay_cached_batches(self) -> Iterable[List[Dict[str, object]]]:
        """Load cached batches from disk for recovery scenarios."""

        for path in sorted(self._cache_dir.glob("batch-*.json")):
            with path.open("r", encoding="utf-8") as handle:
                yield json.load(handle)

    def trim_cache(self, max_files: int = 100) -> None:
        """Limit the number of cached batches on disk."""

        files = sorted(self._cache_dir.glob("batch-*.json"))
        for path in files[:-max_files]:
            path.unlink(missing_ok=True)
>>>> main
