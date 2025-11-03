"""A/B testing utilities."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import List

from .config import ABTestConfig


@dataclass(slots=True)
class ExperimentResult:
    experiment_id: str
    started_at: datetime
    ended_at: datetime
    control_metric: float
    treatment_metric: float
    p_value: float
    significant: bool


class ABTestManager:
    """Manage A/B tests for deployment verification."""

    def __init__(self, config: ABTestConfig) -> None:
        config.validate()
        self._config = config
        self._experiments: List[ExperimentResult] = []

    def run_experiment(self, experiment_id: str, control: List[float], treatment: List[float]) -> ExperimentResult:
        if len(control) < self._config.min_users or len(treatment) < self._config.min_users:
            raise ValueError("Insufficient sample size for experiment")
        mean_control = sum(control) / len(control)
        mean_treatment = sum(treatment) / len(treatment)
        variance_control = self._variance(control, mean_control)
        variance_treatment = self._variance(treatment, mean_treatment)
        p_value = self._welch_ttest(mean_control, mean_treatment, variance_control, variance_treatment, len(control), len(treatment))
        significant = p_value < self._config.significance_level
        result = ExperimentResult(
            experiment_id=experiment_id,
            started_at=datetime.utcnow(),
            ended_at=datetime.utcnow() + self._config.experiment_window,
            control_metric=mean_control,
            treatment_metric=mean_treatment,
            p_value=p_value,
            significant=significant,
        )
        self._experiments.append(result)
        return result

    def history(self) -> List[ExperimentResult]:
        return list(self._experiments)

    def _variance(self, values: List[float], mean: float) -> float:
        return sum((value - mean) ** 2 for value in values) / (len(values) - 1)

    def _welch_ttest(
        self,
        mean_control: float,
        mean_treatment: float,
        var_control: float,
        var_treatment: float,
        n_control: int,
        n_treatment: int,
    ) -> float:
        numerator = mean_treatment - mean_control
        denominator = math.sqrt(var_control / n_control + var_treatment / n_treatment)
        if denominator == 0:
            return 1.0
        t_stat = numerator / denominator
        df_numerator = (var_control / n_control + var_treatment / n_treatment) ** 2
        df_denominator = ((var_control ** 2) / ((n_control ** 2) * (n_control - 1))) + (
            (var_treatment ** 2) / ((n_treatment ** 2) * (n_treatment - 1))
        )
        dof = df_numerator / df_denominator if df_denominator else 1
        return self._student_t_cdf(-abs(t_stat), dof) * 2

    def _student_t_cdf(self, x: float, dof: float) -> float:
        # Use approximation via incomplete beta function expansion
        # to avoid heavy dependencies while keeping monotonic behaviour.
        a = 0.5 * dof
        b = 0.5
        bt = math.exp(
            math.lgamma(a + b)
            - math.lgamma(a)
            - math.lgamma(b)
            + a * math.log(dof)
            - (a + b) * math.log(dof + x * x)
        )
        if x < 0:
            return bt / 2
        return 1 - bt / 2
