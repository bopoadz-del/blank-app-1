"""Deployment orchestration utilities for YOLOv8 TensorRT pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from .config import YOLOv8Config
from .inference import TensorRTInferenceSession
from .profiling import Profiler
from .tensorrt_engine import EngineBuilder, load_engine, save_engine

LOGGER = logging.getLogger(__name__)


class DeploymentManager:
    def __init__(
        self,
        config: YOLOv8Config,
        class_names: List[str],
        engine_initializer: Optional[Callable[[EngineBuilder], bytes]] = None,
    ) -> None:
        self.config = config
        self.class_names = class_names
        self.engine_initializer = engine_initializer
        self.profiler = Profiler(config.profiling)

    def prepare_engine(self) -> bytes:
        engine_path = self.config.deployment.engine
        if engine_path.exists() and self.config.tensorrt.engine_cache:
            LOGGER.info("Loading cached TensorRT engine from %s", engine_path)
            return load_engine(engine_path)

        builder = EngineBuilder(self.config.tensorrt, self.config.quantization)
        if self.engine_initializer:
            engine_bytes = self.engine_initializer(builder)
        else:
            engine_bytes = builder.build(lambda b, n, c: None)
        save_engine(engine_bytes, engine_path)
        return engine_bytes

    def create_session(self) -> TensorRTInferenceSession:
        engine_bytes = self.prepare_engine()
        session = TensorRTInferenceSession(
            self.config,
            engine_loader=lambda: engine_bytes,
            class_names=self.class_names,
        )
        session.load_engine()
        return session

    def export_config(self, path: Path) -> None:
        data = self.config.as_dict()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))


__all__ = ["DeploymentManager"]
