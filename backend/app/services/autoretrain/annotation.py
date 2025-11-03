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
