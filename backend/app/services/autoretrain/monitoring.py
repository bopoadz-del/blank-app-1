"""In-memory monitoring helpers."""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, List, MutableMapping

from .config import MonitoringConfig


class MonitoringDashboard:
    """Track metrics in a sliding window for quick debugging."""

    def __init__(self, config: MonitoringConfig) -> None:
        self._config = config
        self._series: MutableMapping[str, Deque[float]] = defaultdict(deque)

    def record(self, metric: str, value: float) -> None:
        window = self._series[metric]
        window.append(float(value))
        while len(window) > self._config.window:
            window.popleft()

    def latest(self, metric: str) -> float | None:
        window = self._series.get(metric)
        if not window:
            return None
        return window[-1]

    def snapshot(self) -> Dict[str, List[float]]:
        return {name: list(values) for name, values in self._series.items()}
