"""Profiling utilities for the YOLOv8 TensorRT stack."""

from __future__ import annotations

import contextlib
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from .config import ProfilingConfig

try:  # pragma: no cover
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore


@dataclass
class ProfileResult:
    name: str
    durations: List[float]

    def summary(self) -> Dict[str, float]:
        if not self.durations:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": statistics.mean(self.durations),
            "std": statistics.pstdev(self.durations) if len(self.durations) > 1 else 0.0,
            "min": min(self.durations),
            "max": max(self.durations),
        }


class Profiler:
    def __init__(self, config: ProfilingConfig) -> None:
        self.config = config
        self.results: Dict[str, ProfileResult] = {}

    @contextlib.contextmanager
    def record(self, name: str) -> Iterable[None]:
        if not self.config.enable:
            yield
            return
        start = time.perf_counter()
        yield
        duration = time.perf_counter() - start
        self.results.setdefault(name, ProfileResult(name, [])).durations.append(duration)

    def warmup(self, fn: Callable[[], None]) -> None:
        for _ in range(self.config.warmup_runs):
            fn()

    def measure(self, fn: Callable[[], None]) -> None:
        for _ in range(self.config.measurement_runs):
            fn()

    def export(self, path: Optional[Path] = None) -> None:
        if not self.config.export_trace:
            return
        target = path or self.config.export_path
        if target is None:
            return
        lines = ["name,mean,std,min,max"]
        for result in self.results.values():
            stats = result.summary()
            lines.append(
                f"{result.name},{stats['mean']:.6f},{stats['std']:.6f},{stats['min']:.6f},{stats['max']:.6f}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines))


__all__ = ["Profiler", "ProfileResult"]
