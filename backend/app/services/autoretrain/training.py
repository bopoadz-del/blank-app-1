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
