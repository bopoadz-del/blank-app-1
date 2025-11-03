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
