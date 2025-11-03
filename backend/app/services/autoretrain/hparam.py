"""Hyperparameter tuning utilities."""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Tuple

from .config import HyperParameterConfig


@dataclass(slots=True)
class TrialResult:
    params: Dict[str, object]
    score: float


class HyperParameterSearch:
    """Simple exhaustive/grid search implementation."""

    def __init__(self, config: HyperParameterConfig) -> None:
        config.validate()
        self._config = config
        self._trials: List[TrialResult] = []

    @property
    def trials(self) -> List[TrialResult]:
        return list(self._trials)

    def candidates(self) -> Iterator[Dict[str, object]]:
        keys = list(self._config.search_space.keys())
        values = [self._config.search_space[key] for key in keys]
        for combo in itertools.product(*values):
            yield dict(zip(keys, combo))

    def run(self, evaluator: Iterable[Tuple[Dict[str, object], float]]) -> TrialResult:
        best = TrialResult(params={}, score=float("-inf"))
        for params, score in evaluator:
            trial = TrialResult(params=params, score=score)
            self._trials.append(trial)
            if score > best.score:
                best = trial
            if len(self._trials) >= self._config.max_trials:
                break
        return best

    def simulate(self) -> TrialResult:
        evaluator = []
        for params in self.candidates():
            score = random.random()
            evaluator.append((params, score))
            if len(evaluator) >= self._config.max_trials:
                break
        return self.run(evaluator)
