"""Deployment orchestration for RT-DETR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import logging
import threading

from .config import RTDETRConfig
from .errors import RTDETRBuilderError, RTDETRError
from .model import ARCHITECTURES, create_architecture_from_config
from .tensorrt_builder import EngineBuildResult, build_engine
from .inference import HotReloadWatcher, RTDETRInferenceSession, create_inference_session

LOGGER = logging.getLogger(__name__)


@dataclass
class RTDETRDeploymentManager:
    config: RTDETRConfig
    session: Optional[RTDETRInferenceSession] = None
    engine: Optional[EngineBuildResult] = None
    watcher: Optional[HotReloadWatcher] = None

    def ensure_engine(self) -> EngineBuildResult:
        if self.engine and Path(self.engine.path).exists():
            LOGGER.debug("Using cached TensorRT engine at %s", self.engine.path)
            return self.engine
        LOGGER.info("Building TensorRT engine for RT-DETR")
        architecture = self._resolve_architecture()
        self.engine = build_engine(self.config, architecture)
        return self.engine

    def ensure_session(self) -> RTDETRInferenceSession:
        if self.session:
            LOGGER.debug("Reusing existing RT-DETR inference session")
            return self.session
        engine = self.ensure_engine()
        self.session = create_inference_session(self.config, engine)
        return self.session

    def start_hot_reload(self) -> None:
        if not self.config.deployment.hot_reload:
            LOGGER.debug("Hot reload disabled in configuration")
            return
        engine_path = self.config.deployment.engine_cache_dir / self.config.deployment.engine_filename
        self.watcher = HotReloadWatcher(self.ensure_session, engine_path, self.config.deployment.watchdog_interval_s)
        self.watcher.start()

    def shutdown(self) -> None:
        if self.watcher:
            self.watcher.stop()
            self.watcher = None
        self.session = None
        self.engine = None

    def _resolve_architecture(self):
        model_name = self.config.model.backbone
        try:
            architecture = ARCHITECTURES.get(model_name)
        except KeyError:
            LOGGER.info("Falling back to configuration-defined architecture for %s", model_name)
            architecture = create_architecture_from_config({"name": model_name})
        return architecture
