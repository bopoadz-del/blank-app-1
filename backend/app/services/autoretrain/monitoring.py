<<<<< codex/debug-application-issues-vqhnb4
"""In-memory monitoring helpers."""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, List, MutableMapping
=======
"""Monitoring dashboard utilities."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Dict, List, Tuple
>>>>> main

from .config import MonitoringConfig


<<<< codex/debug-application-issues-vqhnb4
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
=======
@dataclass(slots=True)
class MetricPoint:
    name: str
    value: float
    timestamp: datetime


class MonitoringDashboard:
    """Aggregate metrics and emit alerts for retraining health."""

    def __init__(self, config: MonitoringConfig) -> None:
        config.validate()
        self._config = config
        self._metrics: Dict[str, Deque[MetricPoint]] = defaultdict(lambda: deque())
        self._alerts: List[Tuple[str, float, datetime]] = []

    def record(self, name: str, value: float, timestamp: datetime | None = None) -> None:
        point = MetricPoint(name=name, value=value, timestamp=timestamp or datetime.utcnow())
        series = self._metrics[name]
        series.append(point)
        self._trim_series(series)
        threshold = self._config.alert_thresholds.get(name)
        if threshold is not None and value > threshold:
            self._alerts.append((name, value, point.timestamp))

    def snapshot(self) -> Dict[str, List[MetricPoint]]:
        return {name: list(points) for name, points in self._metrics.items()}

    def alerts(self) -> List[Tuple[str, float, datetime]]:
        return list(self._alerts)

    def _trim_series(self, series: Deque[MetricPoint]) -> None:
        window_start = datetime.utcnow() - self._config.retention_period
        while series and series[0].timestamp < window_start:
            series.popleft()
>>>> main
