"""RT-DETR high-performance detection service package.

This module exposes user-friendly constructors that assemble the
configurations, engine builders, and inference sessions required
for integrating the RT-DETR TensorRT deployment pipeline.
"""

from .config import RTDETRConfig, RTDETRDatasetConfig, RTDETRModelConfig
from .inference import RTDETRService, RTDETRInferenceSession
from .deployment import RTDETRDeploymentManager
from .profiling import RTDETRProfiler
from .streaming import MultiStreamOrchestrator
from .batching import BatchScheduler
from .benchmarking import benchmark, BenchmarkResult
from .optimization import (
    OptimizationPass,
    OptimizationPipeline,
    default_optimization_pipeline,
    LatencyOptimizer,
)
from .errors import RTDETRError, RTDETRConfigurationError, RTDETRExecutionError

__all__ = [
    "RTDETRConfig",
    "RTDETRDatasetConfig",
    "RTDETRModelConfig",
    "RTDETRService",
    "RTDETRInferenceSession",
    "RTDETRDeploymentManager",
    "RTDETRProfiler",
    "MultiStreamOrchestrator",
    "BatchScheduler",
    "BenchmarkResult",
    "benchmark",
    "OptimizationPass",
    "OptimizationPipeline",
    "default_optimization_pipeline",
    "LatencyOptimizer",
    "RTDETRError",
    "RTDETRConfigurationError",
    "RTDETRExecutionError",
]
