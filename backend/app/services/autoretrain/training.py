<<<<< codex/debug-application-issues-vqhnb4
"""Training orchestration for a simple logistic regression model."""
from __future__ import annotations

import math
from datetime import datetime
from random import Random
from typing import Dict, Iterable, List, Tuple

from .config import TrainingConfig
from .entities import AnnotatedSample, DatasetSplit, TrainingArtifact
from .hparam import HyperParameterSearch
from .evaluation import EvaluationSuite


class LogisticRegressionModel:
    """Minimal logistic regression implementation using batch gradient descent."""

    def __init__(self, n_features: int) -> None:
        self.weights: List[float] = [0.0] * n_features
        self.bias: float = 0.0

    def predict_proba(self, features: Iterable[float]) -> float:
        z = self.bias
        for weight, value in zip(self.weights, features):
            z += weight * value
        # Clamp to avoid overflow in exp
        z = max(min(z, 60.0), -60.0)
        return 1.0 / (1.0 + math.exp(-z))


class TrainingOrchestrator:
    """Train and select the best model using the configured search space."""

    def __init__(
        self,
        config: TrainingConfig,
        search: HyperParameterSearch,
        *,
        seed: int = 23,
    ) -> None:
        self._config = config
        self._search = search
        self._random = Random(seed)

    def train(self, split: DatasetSplit, evaluator: EvaluationSuite) -> Tuple[TrainingArtifact, Dict[str, float]]:
        if not split.train:
            raise ValueError("Training split is empty; unable to fit a model")

        n_features = len(split.train[0].features)
        candidates = list(self._search.generate())
        if not candidates:
            candidates = [
                {
                    "learning_rate": self._config.learning_rate,
                    "l2_strength": self._config.l2_strength,
                    "batch_size": self._config.batch_size,
                }
            ]

        best_artifact: TrainingArtifact | None = None
        best_metrics: Dict[str, float] | None = None
        best_score = -math.inf

        for params in candidates:
            model = LogisticRegressionModel(n_features)
            epochs, final_loss = self._fit_model(model, split.train, params)
            artifact = TrainingArtifact(
                version=datetime.utcnow().strftime("%Y%m%d%H%M%S"),
                params={
                    "learning_rate": params["learning_rate"],
                    "l2_strength": params["l2_strength"],
                    "batch_size": float(params["batch_size"]),
                    "epochs": float(epochs),
                    "final_loss": final_loss,
                },
                weights=list(model.weights),
                bias=model.bias,
            )
            validation_set = split.val or split.train
            metrics = evaluator.evaluate(validation_set, artifact)
            score = self._score(metrics)
            if score > best_score:
                best_artifact = artifact
                best_metrics = metrics
                best_score = score

        assert best_artifact is not None and best_metrics is not None  # defensive
        return best_artifact, best_metrics

    def _fit_model(self, model: LogisticRegressionModel, samples: List[AnnotatedSample], params: Dict[str, float]) -> Tuple[int, float]:
        learning_rate = float(params["learning_rate"])
        l2_strength = float(params["l2_strength"])
        batch_size = max(1, int(params["batch_size"]))

        history_loss = 0.0
        for epoch in range(1, self._config.max_epochs + 1):
            self._random.shuffle(samples)
            total_loss = 0.0
            for start in range(0, len(samples), batch_size):
                batch = samples[start : start + batch_size]
                gradients = [0.0] * len(model.weights)
                bias_grad = 0.0
                for sample in batch:
                    prediction = model.predict_proba(sample.features)
                    error = prediction - sample.label
                    for idx, value in enumerate(sample.features):
                        gradients[idx] += error * value
                    bias_grad += error
                    total_loss += self._binary_cross_entropy(prediction, sample.label)
                if batch:
                    scale = 1.0 / len(batch)
                    for idx, grad in enumerate(gradients):
                        grad = grad * scale + l2_strength * model.weights[idx]
                        model.weights[idx] -= learning_rate * grad
                    model.bias -= learning_rate * (bias_grad * scale)
            average_loss = total_loss / len(samples)
            if abs(history_loss - average_loss) < self._config.tolerance:
                return epoch, average_loss
            history_loss = average_loss
        return self._config.max_epochs, history_loss

    @staticmethod
    def _binary_cross_entropy(prediction: float, target: int) -> float:
        prediction = min(max(prediction, 1e-8), 1 - 1e-8)
        return -(
            target * math.log(prediction)
            + (1 - target) * math.log(1 - prediction)
        )

    @staticmethod
    def _score(metrics: Dict[str, float]) -> float:
        if "f1" in metrics:
            return metrics["f1"]
        if "accuracy" in metrics:
            return metrics["accuracy"]
        return sum(metrics.values()) / max(len(metrics), 1)
=======
"""Training orchestration utilities."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .config import TrainingConfig
from .hparam import HyperParameterSearch, TrialResult


@dataclass(slots=True)
class TrainingArtifact:
    version: str
    metrics: Dict[str, float]
    params: Dict[str, object]


class TrainingOrchestrator:
    """Coordinate training runs and hyperparameter search."""

    def __init__(self, config: TrainingConfig, search: Optional[HyperParameterSearch] = None) -> None:
        config.validate()
        self._config = config
        self._search = search
        self._history: List[TrainingArtifact] = []

    def train(self, dataset: Dict[str, Iterable[Dict[str, object]]]) -> TrainingArtifact:
        """Run a training job, optionally using hyperparameter search."""

        best_trial: Optional[TrialResult] = None
        if self._search:
            evaluator = []
            for params in self._search.candidates():
                score = self._simulate_training_run(params, dataset)
                evaluator.append((params, score))
                if len(evaluator) >= self._search._config.max_trials:
                    break
            best_trial = self._search.run(evaluator)
        params = best_trial.params if best_trial else {"learning_rate": 0.001}
        metrics = {
            "loss": random.uniform(0.1, 0.5),
            "accuracy": random.uniform(0.7, 0.99),
        }
        artifact = TrainingArtifact(version=self._create_checkpoint(params, metrics), metrics=metrics, params=params)
        self._history.append(artifact)
        return artifact

    def _simulate_training_run(self, params: Dict[str, object], dataset: Dict[str, Iterable[Dict[str, object]]]) -> float:
        random.seed(hash(frozenset(params.items())))
        size = sum(1 for split in dataset.values() for _ in split)
        return random.random() + min(1.0, size / 10000)

    def _create_checkpoint(self, params: Dict[str, object], metrics: Dict[str, float]) -> str:
        checkpoint = f"ckpt-{len(self._history)+1:04d}"
        return checkpoint

    def history(self) -> List[TrainingArtifact]:
        return list(self._history)
>>>>> main
