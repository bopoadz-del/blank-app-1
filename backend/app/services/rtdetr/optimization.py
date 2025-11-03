"""Inference optimization utilities for RT-DETR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import logging
import time

import numpy as np

from .config import RTDETRConfig
from .errors import RTDETRExecutionError

LOGGER = logging.getLogger(__name__)


@dataclass
class OptimizationPass:
    name: str
    description: str
    enabled: bool = True
    parameters: Dict[str, object] = field(default_factory=dict)

    def apply(self, tensor: np.ndarray) -> np.ndarray:
        LOGGER.debug("Applying optimization pass %s", self.name)
        if not self.enabled:
            return tensor
        if self.name == "quantize":
            scale = float(self.parameters.get("scale", 0.02))
            zero_point = float(self.parameters.get("zero_point", 0.0))
            tensor = np.clip(np.round(tensor / scale + zero_point), -128, 127)
        elif self.name == "dequantize":
            scale = float(self.parameters.get("scale", 0.02))
            zero_point = float(self.parameters.get("zero_point", 0.0))
            tensor = (tensor.astype(np.float32) - zero_point) * scale
        elif self.name == "fuse_bias":
            bias = float(self.parameters.get("bias", 0.0))
            tensor = tensor + bias
        return tensor


@dataclass
class OptimizationPipeline:
    passes: List[OptimizationPass] = field(default_factory=list)

    def run(self, tensor: np.ndarray) -> np.ndarray:
        LOGGER.debug("Running optimization pipeline with %s passes", len(self.passes))
        for opt_pass in self.passes:
            tensor = opt_pass.apply(tensor)
        return tensor

    def add_pass(self, opt_pass: OptimizationPass) -> None:
        LOGGER.debug("Adding optimization pass %s", opt_pass.name)
        self.passes.append(opt_pass)


def default_optimization_pipeline(config: RTDETRConfig) -> OptimizationPipeline:
    pipeline = OptimizationPipeline()
    if config.precision.precision == "int8":
        pipeline.add_pass(OptimizationPass(name="quantize", description="Quantize tensors", parameters={"scale": 0.02}))
        pipeline.add_pass(OptimizationPass(name="dequantize", description="Restore tensors", parameters={"scale": 0.02}))
    pipeline.add_pass(OptimizationPass(name="fuse_bias", description="Fuse bias into outputs", parameters={"bias": 0.0}))
    return pipeline


@dataclass
class LatencyOptimizer:
    config: RTDETRConfig
    target_latency_ms: float = 4.0
    step_size: float = 0.5
    max_iterations: int = 20

    def tune(self, runner) -> Dict[str, float]:
        LOGGER.info("Starting latency optimization targeting %.2fms", self.target_latency_ms)
        best_latency = float("inf")
        best_settings: Dict[str, float] = {}
        for iteration in range(self.max_iterations):
            start = time.perf_counter()
            runner()
            latency = (time.perf_counter() - start) * 1000
            LOGGER.debug("Iteration %s latency=%.3fms", iteration, latency)
            if latency < best_latency:
                best_latency = latency
                best_settings = {"iteration": iteration, "latency_ms": latency}
            if latency <= self.target_latency_ms:
                LOGGER.info("Target latency achieved at iteration %s", iteration)
                break
        return best_settings


__all__ = [
    "OptimizationPass",
    "OptimizationPipeline",
    "default_optimization_pipeline",
    "LatencyOptimizer",
]
