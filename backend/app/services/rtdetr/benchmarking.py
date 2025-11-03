"""Benchmarking helpers for RT-DETR throughput evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List

import logging
import statistics
import time

import numpy as np

from .profiling import RTDETRProfiler

LOGGER = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    fps: float
    latency_ms: float
    samples: int
    durations: List[float]


def benchmark(
    images: Iterable[np.ndarray],
    run_inference: Callable[[List[np.ndarray]], None],
    *,
    warmup: int = 5,
    repetitions: int = 50,
    batch_size: int = 1,
    profiler: RTDETRProfiler | None = None,
) -> BenchmarkResult:
    LOGGER.info("Starting RT-DETR benchmark: warmup=%s repetitions=%s batch_size=%s", warmup, repetitions, batch_size)
    images = list(images)
    if not images:
        raise ValueError("No images provided for benchmarking")
    batches = [images[i : i + batch_size] for i in range(0, len(images), batch_size)]
    for _ in range(warmup):
        for batch in batches:
            run_inference(batch)
    durations: List[float] = []
    for _ in range(repetitions):
        for batch in batches:
            start = time.perf_counter()
            run_inference(batch)
            duration = time.perf_counter() - start
            durations.append(duration)
            if profiler:
                profiler.measure("benchmark_iteration", duration)
    if not durations:
        raise RuntimeError("Benchmark produced no timings")
    avg_duration = statistics.fmean(durations)
    fps = batch_size / avg_duration if avg_duration > 0 else 0.0
    latency_ms = avg_duration * 1000
    LOGGER.info("Benchmark completed: fps=%.2f latency=%.2fms", fps, latency_ms)
    return BenchmarkResult(fps=fps, latency_ms=latency_ms, samples=len(durations), durations=durations)


__all__ = ["benchmark", "BenchmarkResult"]
