"""Evaluation helpers for the automated retraining pipeline."""
from __future__ import annotations

import math
from typing import Dict, Iterable, List

from .config import EvaluationConfig
from .entities import AnnotatedSample, TrainingArtifact


class EvaluationSuite:
    """Compute standard classification metrics."""

    def __init__(self, config: EvaluationConfig) -> None:
        self._config = config

    def evaluate(self, samples: Iterable[AnnotatedSample], artifact: TrainingArtifact) -> Dict[str, float]:
        samples = list(samples)
        if not samples:
            return {metric: 0.0 for metric in self._config.metrics}

        predictions = [
            self._predict_label(sample.features, artifact)
            for sample in samples
        ]
        labels = [sample.label for sample in samples]
        metrics: Dict[str, float] = {}
        metric_set = set(self._config.metrics)

        if "accuracy" in metric_set:
            metrics["accuracy"] = self._accuracy(labels, predictions)

        precision_value = None
        if "precision" in metric_set or "f1" in metric_set:
            precision_value = self._precision(labels, predictions)
            if "precision" in metric_set:
                metrics["precision"] = precision_value

        recall_value = None
        if "recall" in metric_set or "f1" in metric_set:
            recall_value = self._recall(labels, predictions)
            if "recall" in metric_set:
                metrics["recall"] = recall_value

        if "f1" in metric_set:
            if precision_value is None:
                precision_value = self._precision(labels, predictions)
            if recall_value is None:
                recall_value = self._recall(labels, predictions)
            metrics["f1"] = self._f1_score(precision_value, recall_value)

        return metrics

    def validate(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        return {
            metric: metrics.get(metric, 0.0) >= threshold
            for metric, threshold in self._config.thresholds.items()
        }

    def _predict_label(self, features: Iterable[float], artifact: TrainingArtifact) -> int:
        score = artifact.bias
        for weight, value in zip(artifact.weights, features):
            score += weight * value
        score = max(min(score, 60.0), -60.0)
        probability = 1.0 / (1.0 + math.exp(-score))
        return 1 if probability >= 0.5 else 0

    def _accuracy(self, labels: List[int], predictions: List[int]) -> float:
        correct = sum(1 for truth, pred in zip(labels, predictions) if truth == pred)
        return correct / len(labels)

    def _precision(self, labels: List[int], predictions: List[int]) -> float:
        positive = self._config.positive_label
        true_positive = sum(1 for truth, pred in zip(labels, predictions) if pred == positive and truth == positive)
        predicted_positive = sum(1 for pred in predictions if pred == positive)
        if predicted_positive == 0:
            return 0.0
        return true_positive / predicted_positive

    def _recall(self, labels: List[int], predictions: List[int]) -> float:
        positive = self._config.positive_label
        true_positive = sum(1 for truth, pred in zip(labels, predictions) if truth == positive and pred == positive)
        actual_positive = sum(1 for truth in labels if truth == positive)
        if actual_positive == 0:
            return 0.0
        return true_positive / actual_positive

    @staticmethod
    def _f1_score(precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
