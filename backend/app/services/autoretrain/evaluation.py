"""Evaluation utilities for retrained models."""
from __future__ import annotations

from typing import Dict, Iterable

from .config import EvaluationConfig


class EvaluationSuite:
    """Compute evaluation metrics and validate thresholds."""

    def __init__(self, config: EvaluationConfig) -> None:
        config.validate()
        self._config = config

    def evaluate(self, predictions: Iterable[Dict[str, float]]) -> Dict[str, float]:
        cache = list(predictions)
        metrics: Dict[str, float] = {}
        for metric in self._config.metrics:
            method = getattr(self, f"_metric_{metric}", None)
            if not method:
                raise ValueError(f"Unsupported metric: {metric}")
            metrics[metric] = method(cache)
        return metrics

    def validate(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        results = {}
        for name, threshold in self._config.thresholds.items():
            value = metrics.get(name)
            if value is None:
                results[name] = False
            else:
                results[name] = value >= threshold
        return results

    def _metric_accuracy(self, predictions: Iterable[Dict[str, float]]) -> float:
        total = 0
        correct = 0
        for item in predictions:
            total += 1
            if item.get("label") == item.get("prediction"):
                correct += 1
        return correct / total if total else 0.0

    def _metric_precision(self, predictions: Iterable[Dict[str, float]]) -> float:
        tp = 0
        fp = 0
        for item in predictions:
            pred = item.get("prediction")
            label = item.get("label")
            if pred == label == 1:
                tp += 1
            elif pred == 1 and label != pred:
                fp += 1
        total = tp + fp
        return tp / total if total else 0.0

    def _metric_recall(self, predictions: Iterable[Dict[str, float]]) -> float:
        tp = 0
        fn = 0
        for item in predictions:
            pred = item.get("prediction")
            label = item.get("label")
            if pred == label == 1:
                tp += 1
            elif label == 1 and pred != label:
                fn += 1
        total = tp + fn
        return tp / total if total else 0.0

    def _metric_f1(self, predictions: Iterable[Dict[str, float]]) -> float:
        cache = list(predictions)
        precision = self._metric_precision(cache)
        recall = self._metric_recall(cache)
        total = precision + recall
        return 2 * precision * recall / total if total else 0.0
