"""TensorRT engine builder scaffolding for RT-DETR."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import logging
import time

from .config import RTDETRConfig
from .errors import (
    RTDETRBuilderError,
    RTDETRCalibrationError,
    RTDETRConfigurationError,
)
from .model import RTDETRArchitecture, architecture_to_trt_layers

LOGGER = logging.getLogger(__name__)


@dataclass
class CalibrationDataset:
    """Simple calibration dataset that yields image paths."""

    images: List[Path] = field(default_factory=list)

    def add_image(self, path: Path) -> None:
        LOGGER.debug("Adding calibration image: %s", path)
        self.images.append(path)

    def __iter__(self) -> Iterable[Path]:
        return iter(self.images)

    def __len__(self) -> int:
        return len(self.images)


@dataclass
class EngineBuildResult:
    path: Path
    precision: str
    build_time_s: float
    layer_count: int


class TensorRTEngineBuilder:
    """Constructs TensorRT engines for the RT-DETR model."""

    def __init__(self, config: RTDETRConfig, architecture: RTDETRArchitecture) -> None:
        self.config = config
        self.architecture = architecture

    def build(self) -> EngineBuildResult:
        LOGGER.info("Starting TensorRT build for RT-DETR")
        if self.config.precision.precision == "int8" and not self._has_calibration_data():
            raise RTDETRCalibrationError(
                "INT8 precision requested but calibration dataset is empty",
                hint="Provide calibration images via config.dataset.calibration_images",
            )
        start = time.perf_counter()
        try:
            layers = architecture_to_trt_layers(self.architecture)
            self._simulate_builder(layers)
        except RTDETRConfigurationError:
            raise
        except Exception as exc:  # pragma: no cover - defensive guard
            raise RTDETRBuilderError("Unexpected failure during TensorRT build", hint=str(exc)) from exc
        duration = time.perf_counter() - start
        engine_path = self.config.deployment.engine_cache_dir / self.config.deployment.engine_filename
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        engine_path.write_text("# simulated TensorRT engine for RT-DETR\n")
        LOGGER.info("TensorRT build completed in %.2fs -> %s", duration, engine_path)
        return EngineBuildResult(
            path=engine_path,
            precision=self.config.precision.precision,
            build_time_s=duration,
            layer_count=len(layers),
        )

    def _simulate_builder(self, layers: List[Dict[str, object]]) -> None:
        LOGGER.debug("Simulating TensorRT builder with %s layers", len(layers))
        if not layers:
            raise RTDETRConfigurationError("No layers defined for TensorRT engine")
        for index, layer in enumerate(layers):
            LOGGER.debug("Building layer %s: %s", index, layer)
            time.sleep(0.001)  # simulate work

    def _has_calibration_data(self) -> bool:
        if self.config.dataset.calibration_images:
            return True
        default_image_dir = Path(self.config.dataset.root) / "calibration"
        return default_image_dir.exists()


def build_engine(config: RTDETRConfig, architecture: RTDETRArchitecture) -> EngineBuildResult:
    builder = TensorRTEngineBuilder(config, architecture)
    return builder.build()
