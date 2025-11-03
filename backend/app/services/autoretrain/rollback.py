"""Rollback helpers for trained model versions."""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Iterable, Mapping, Optional, Tuple

from .config import RollbackConfig


class RollbackManager:
    """Keep track of recently trained models for quick recovery."""

    def __init__(self, config: RollbackConfig) -> None:
        self._config = config
        self._registry: Deque[Tuple[str, Dict[str, float]]] = deque()

    def register(self, version: str, metrics: Mapping[str, float]) -> None:
        self._registry.appendleft((version, dict(metrics)))
        while len(self._registry) > self._config.max_versions:
            self._registry.pop()

    def best_candidate(self, metric: str) -> Optional[Tuple[str, Dict[str, float]]]:
        best_score = float("-inf")
        best_entry: Optional[Tuple[str, Dict[str, float]]] = None
        for entry in self._registry:
            score = entry[1].get(metric)
            if score is None:
                continue
            if score > best_score:
                best_score = score
                best_entry = entry
        return best_entry

    def history(self) -> Iterable[Tuple[str, Dict[str, float]]]:
        return tuple(self._registry)
