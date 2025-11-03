<<<<<codex/debug-application-issues-vqhnb4
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
=======
"""Annotation pipeline for automated retraining."""
from __future__ import annotations

import statistics
from collections import Counter
from typing import Dict, Iterable, List, Sequence

from .config import AnnotationConfig


class AnnotationPipeline:
    """Runs a hybrid human/automatic annotation pipeline."""

    def __init__(self, config: AnnotationConfig) -> None:
        config.validate()
        self._config = config

    def annotate(self, records: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
        """Annotate records using automatic heuristics and consensus."""

        annotated = []
        for record in records:
            machine_labels = self._apply_rules(record)
            human_votes = self._simulate_labelers(machine_labels)
            label, confidence = self._consensus_label(human_votes)
            annotated.append({**record, "label": label, "confidence": confidence})
        return annotated

    def _apply_rules(self, record: Dict[str, object]) -> Counter:
        payload = record.get("payload", {})
        counter: Counter = Counter()
        for label, rule in self._config.auto_label_rules.items():
            score = 0.0
            for feature, weight in rule.items():
                score += float(payload.get(feature, 0.0)) * weight
            if score > 0.5:
                counter[label] += 1
        return counter

    def _simulate_labelers(self, machine_votes: Counter) -> Counter:
        votes = Counter(machine_votes)
        for labeler, accuracy in self._config.labelers.items():
            preferred = labeler.split(":")[-1]
            if machine_votes:
                top_label, _ = machine_votes.most_common(1)[0]
            else:
                top_label = preferred
            agreed = statistics.fmean([accuracy, 0.5]) > 0.5
            votes[top_label if agreed else preferred] += 1
        return votes

    def _consensus_label(self, votes: Counter) -> tuple[str, float]:
        if not votes:
            return "unlabeled", 0.0
        label, count = votes.most_common(1)[0]
        total = sum(votes.values())
        confidence = count / total if total else 0.0
        if confidence < self._config.consensus_threshold:
            return "review", confidence
        return label, confidence

    def sample_for_review(self, annotated: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
        """Sample annotated records for human review."""

        annotated_list = list(annotated)
        sample_size = max(1, int(len(annotated_list) * self._config.review_sample_rate))
        return annotated_list[:sample_size]
>>>>> main
