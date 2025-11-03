"""Hyperparameter search helpers."""
from __future__ import annotations

from itertools import product
from typing import Dict, Iterable

from .config import HyperParameterConfig


class HyperParameterSearch:
    """Enumerate parameter combinations according to the configuration."""

    def __init__(self, config: HyperParameterConfig) -> None:
        self._config = config

    def generate(self) -> Iterable[Dict[str, float]]:
        combos = product(
            self._config.learning_rates,
            self._config.l2_strengths,
            self._config.batch_sizes,
        )
        for index, (lr, l2, batch) in enumerate(combos):
            if self._config.max_trials is not None and index >= self._config.max_trials:
                break
            yield {
                "learning_rate": float(lr),
                "l2_strength": float(l2),
                "batch_size": int(batch),
            }
