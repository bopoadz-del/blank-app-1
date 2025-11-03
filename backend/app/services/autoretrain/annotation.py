"""Annotation pipeline helpers."""
from __future__ import annotations

from random import Random
from typing import Iterable, List

from .config import AnnotationConfig
from .entities import AnnotatedSample, RawSample


class AnnotationPipeline:
    """Applies simple consensus rules to raw samples."""

    def __init__(self, config: AnnotationConfig, *, seed: int = 17) -> None:
        self._config = config
        self._random = Random(seed)
        total = sum(config.labelers.values())
        self._normaliser = 1.0 / total if total else 0.0

    def annotate(self, samples: Iterable[RawSample]) -> List[AnnotatedSample]:
        annotated: List[AnnotatedSample] = []
        for sample in samples:
            label, consensus = self._determine_label(sample)
            metadata = dict(sample.metadata)
            metadata["consensus_score"] = consensus
            metadata["requires_review"] = self._random.random() < self._config.review_sample_rate
            annotated.append(AnnotatedSample(features=sample.features, label=label, metadata=metadata))
        return annotated

    def _determine_label(self, sample: RawSample) -> tuple[int, float]:
        if sample.label is not None:
            return int(sample.label), float(sample.label)

        weighted_votes = 0.0
        for name, weight in self._config.labelers.items():
            rule_threshold = self._config.auto_label_rules.get(name, 0.5)
            signal = sample.metadata.get(name, 0.0)
            if signal >= rule_threshold:
                weighted_votes += weight

        consensus = weighted_votes * self._normaliser
        label = 1 if consensus >= self._config.consensus_threshold else 0
        return label, consensus
