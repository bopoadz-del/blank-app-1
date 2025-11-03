"""TensorRT engine builder and serializer for YOLOv8."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterable, List, Optional

from .config import QuantizationConfig, TensorRTBuildConfig

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import tensorrt as trt  # type: ignore
except Exception:  # pragma: no cover - fallback when TensorRT unavailable
    trt = None  # type: ignore


class DummyNetwork:  # pragma: no cover - fallback stub
    def __init__(self) -> None:
        self.layers: List[str] = []

    def add_input(self, name: str, dtype: str, shape: tuple) -> None:
        self.layers.append(f"Input({name}, {dtype}, {shape})")

    def mark_output(self, tensor) -> None:
        self.layers.append(f"Output({tensor})")


class Int8Calibrator:
    def __init__(self, config: QuantizationConfig, batch_loader: Callable[[], Iterable[bytes]]) -> None:
        self.config = config
        self.batch_loader = batch_loader
        self.cache_path = config.calibrator_cache
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

    def get_batch_size(self) -> int:
        return 1

    def get_batch(self, names) -> Optional[List[bytes]]:  # pragma: no cover - runtime only
        try:
            batch = next(self._iterator)
        except StopIteration:
            return None
        return batch

    def read_calibration_cache(self) -> Optional[bytes]:  # pragma: no cover
        if self.cache_path.exists() and self.config.load_existing:
            LOGGER.info("Loading INT8 calibration cache from %s", self.cache_path)
            return self.cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:  # pragma: no cover
        LOGGER.info("Writing INT8 calibration cache to %s", self.cache_path)
        self.cache_path.write_bytes(cache)

    def __enter__(self):  # pragma: no cover
        self._iterator = iter(self.batch_loader())
        return self

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover
        self._iterator = None


class EngineBuilder:
    def __init__(
        self,
        build_config: TensorRTBuildConfig,
        quant_config: Optional[QuantizationConfig] = None,
        logger: Optional["trt.Logger"] = None,
    ) -> None:
        self.build_config = build_config
        self.quant_config = quant_config
        if trt is not None:
            self.logger = logger or trt.Logger(trt.Logger.WARNING)
        else:
            self.logger = None

    def _create_builder(self):  # pragma: no cover - runtime only
        if trt is None:
            raise RuntimeError("TensorRT is not available")
        return trt.Builder(self.logger)

    def _create_network(self, builder):  # pragma: no cover
        if trt is None:
            return DummyNetwork()
        return builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    def build(
        self,
        on_build: Callable[[object, object, object], None],
        batch_loader: Optional[Callable[[], Iterable[bytes]]] = None,
    ) -> bytes:
        if trt is None:
            LOGGER.warning("TensorRT not found; returning placeholder engine bytes")
            return b"DUMMY_ENGINE"

        builder = self._create_builder()
        network = self._create_network(builder)
        config = builder.create_builder_config()
        config.max_workspace_size = self.build_config.max_workspace_size
        config.set_flag(trt.BuilderFlag.FP16) if self.build_config.fp16 else None
        if self.build_config.int8:
            config.set_flag(trt.BuilderFlag.INT8)
            if batch_loader is None:
                raise ValueError("INT8 requested but no batch loader provided")
            calibrator = Int8Calibrator(self.quant_config, batch_loader) if self.quant_config else None
            if calibrator is None:
                raise ValueError("Quantization config required for INT8 builds")
            config.int8_calibrator = calibrator
        if self.build_config.strict_types:
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        if self.build_config.enable_dla:
            config.default_device_type = trt.DeviceType.DLA
            config.DLA_core = self.build_config.dla_core
        if self.build_config.profiling_verbosity:
            config.profiling_verbosity = getattr(trt.ProfilingVerbosity, self.build_config.profiling_verbosity)

        on_build(builder, network, config)
        engine = builder.build_engine(network, config)
        serialized = engine.serialize()
        return bytes(serialized)


def save_engine(engine_bytes: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(engine_bytes)


def load_engine(path: Path) -> bytes:
    return path.read_bytes()


__all__ = [
    "EngineBuilder",
    "save_engine",
    "load_engine",
    "Int8Calibrator",
]
