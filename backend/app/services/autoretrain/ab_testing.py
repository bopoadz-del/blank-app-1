"""Simple A/B test helper."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .config import ABTestConfig


@dataclass(slots=True)
class ABTestResult:
    """Summary of an A/B experiment."""

    experiment_id: str
    control_mean: float
    treatment_mean: float
    p_value: float
    significant: bool


class ABTestManager:
    """Run Welch-style two-sided tests on metric samples."""

    def __init__(self, config: ABTestConfig) -> None:
        self._config = config

    def run_experiment(self, experiment_id: str, control: Iterable[float], treatment: Iterable[float]) -> ABTestResult:
        control_samples = list(control)
        treatment_samples = list(treatment)
        total = len(control_samples) + len(treatment_samples)
        if total < self._config.minimum_total:
            raise ValueError("Not enough samples to run experiment")

        control_mean = self._mean(control_samples)
        treatment_mean = self._mean(treatment_samples)
        control_var = self._variance(control_samples, control_mean)
        treatment_var = self._variance(treatment_samples, treatment_mean)

        se = math.sqrt(
            (control_var / len(control_samples) if len(control_samples) > 0 else 0.0)
            + (treatment_var / len(treatment_samples) if len(treatment_samples) > 0 else 0.0)
        )
        if se == 0:
            p_value = 1.0
        else:
            z = (treatment_mean - control_mean) / se
            p_value = 2 * 0.5 * math.erfc(abs(z) / math.sqrt(2))

        significant = p_value < self._config.significance_level
        return ABTestResult(
            experiment_id=experiment_id,
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            p_value=p_value,
            significant=significant,
        )

    @staticmethod
    def _mean(samples: list[float]) -> float:
        if not samples:
            return 0.0
        return sum(samples) / len(samples)

    @staticmethod
    def _variance(samples: list[float], mean: float) -> float:
        if len(samples) <= 1:
            return 0.0
        return sum((value - mean) ** 2 for value in samples) / (len(samples) - 1)
