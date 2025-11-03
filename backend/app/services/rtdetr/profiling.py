"""Profiling utilities for RT-DETR."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List

import json
import logging
import statistics
import time

from .config import ProfilingConfig
from .errors import RTDETRProfilingError

LOGGER = logging.getLogger(__name__)


@dataclass
class Measurement:
    name: str
    values: List[float] = field(default_factory=list)

    def record(self, value: float) -> None:
        LOGGER.debug("Recording measurement %s=%.6f", self.name, value)
        self.values.append(value)

    def stats(self) -> Dict[str, float]:
        if not self.values:
            return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0}
        sorted_values = sorted(self.values)
        return {
            "count": len(sorted_values),
            "mean": statistics.fmean(sorted_values),
            "p50": statistics.median(sorted_values),
            "p95": sorted_values[int(0.95 * (len(sorted_values) - 1))],
        }


@dataclass
class RTDETRProfiler:
    config: ProfilingConfig
    measurements: Dict[str, Measurement] = field(default_factory=dict)
    _last_report: float = field(default_factory=time.time)

    def measure(self, name: str, value: float) -> None:
        measurement = self.measurements.setdefault(name, Measurement(name))
        measurement.record(value)
        now = time.time()
        if self.config.aggregate_metrics and now - self._last_report >= self.config.min_report_interval_s:
            self.report()
            self._last_report = now

    def report(self) -> Dict[str, Dict[str, float]]:
        LOGGER.info("Reporting RT-DETR profiling metrics")
        return {name: measurement.stats() for name, measurement in self.measurements.items()}

    def dump_timeline(self) -> None:
        if not self.config.enable_profiling or not self.config.capture_timeline:
            return
        timeline = {
            "measurements": {name: measurement.values for name, measurement in self.measurements.items()}
        }
        try:
            self.config.timeline_path.parent.mkdir(parents=True, exist_ok=True)
            self.config.timeline_path.write_text(json.dumps(timeline, indent=2))
        except OSError as exc:  # pragma: no cover - filesystem guard
            raise RTDETRProfilingError("Failed to write profiling timeline", hint=str(exc)) from exc


__all__ = ["RTDETRProfiler", "Measurement"]
