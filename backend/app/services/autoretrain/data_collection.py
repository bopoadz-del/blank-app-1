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
