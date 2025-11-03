"""Dataset management utilities."""
from __future__ import annotations

import json
from collections import deque
from datetime import datetime
from pathlib import Path
from random import Random
from typing import Iterable

from .config import DatasetConfig
from .entities import AnnotatedSample, DatasetSplit


class DatasetRepository:
    """Persist annotated samples and provide deterministic splits."""

    def __init__(self, config: DatasetConfig, *, seed: int = 19) -> None:
        self._config = config
        self._random = Random(seed)
        self._root = config.root_dir
        self._root.mkdir(parents=True, exist_ok=True)
        existing = sorted(path for path in self._root.iterdir() if path.is_dir())
        self._versions: deque[Path] = deque(existing)

    def create_version(self, samples: Iterable[AnnotatedSample]) -> Path:
        version = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        version_dir = self._root / version
        version_dir.mkdir(parents=True, exist_ok=True)
        data_path = version_dir / "dataset.json"
        payload = [
            {"features": sample.features, "label": sample.label, "metadata": sample.metadata}
            for sample in samples
        ]
        data_path.write_text(json.dumps(payload, indent=2))
        self._versions.append(version_dir)
        self._prune_old_versions()
        return version_dir

    def split(self, samples: Iterable[AnnotatedSample]) -> DatasetSplit:
        materialised = list(samples)
        indices = list(range(len(materialised)))
        self._random.shuffle(indices)

        n_total = len(materialised)
        n_train = max(1, int(n_total * self._config.train_ratio)) if n_total else 0
        n_val = max(0, int(n_total * self._config.val_ratio)) if n_total else 0
        n_train = min(n_train, n_total)
        n_val = min(n_val, n_total - n_train)
        n_test = max(0, n_total - n_train - n_val)

        train = [materialised[i] for i in indices[:n_train]]
        val = [materialised[i] for i in indices[n_train : n_train + n_val]]
        test = [materialised[i] for i in indices[n_train + n_val : n_train + n_val + n_test]]
        return DatasetSplit(train=train, val=val, test=test)

    def _prune_old_versions(self) -> None:
        from shutil import rmtree

        while len(self._versions) > self._config.retention:
            old_path = self._versions.popleft()
            rmtree(old_path, ignore_errors=True)
