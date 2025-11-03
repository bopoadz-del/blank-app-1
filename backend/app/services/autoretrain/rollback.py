"""Rollback management for automated retraining."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from .config import RollbackConfig


@dataclass(slots=True)
class ModelVersion:
    version: str
    created_at: datetime
    metrics: Dict[str, float]


class RollbackManager:
    """Track deployed model versions and enable rollback."""

    def __init__(self, config: RollbackConfig) -> None:
        config.validate()
        self._config = config
        self._versions: List[ModelVersion] = []
        self._current: Optional[ModelVersion] = None

    def register(self, version: str, metrics: Dict[str, float]) -> ModelVersion:
        model = ModelVersion(version=version, created_at=datetime.utcnow(), metrics=metrics)
        self._versions.append(model)
        self._current = model
        self._prune()
        return model

    def current(self) -> Optional[ModelVersion]:
        return self._current

    def rollback(self) -> ModelVersion:
        if len(self._versions) < 2:
            raise RuntimeError("No previous versions available for rollback")
        self._versions.pop()
        self._current = self._versions[-1]
        return self._current

    def _prune(self) -> None:
        if len(self._versions) <= self._config.max_versions:
            return
        self._versions = self._versions[-self._config.max_versions :]
