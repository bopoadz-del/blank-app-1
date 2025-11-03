"""Dataset management utilities."""
from __future__ import annotations

import json
import shutil
from collections import deque
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Tuple

from .config import DatasetConfig


class DatasetRepository:
    """Manage dataset versions for automated retraining."""

    def __init__(self, config: DatasetConfig) -> None:
        config.validate()
        self._config = config
        self._versions: Deque[Tuple[str, Path]] = deque()
        self._root = config.root_dir
        self._root.mkdir(parents=True, exist_ok=True)

    def create_version(self, data: Iterable[Dict[str, object]]) -> Path:
        """Create a new dataset version on disk."""

        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        version_dir = self._root / timestamp
        version_dir.mkdir(parents=True, exist_ok=True)
        path = version_dir / "dataset.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(list(data), handle)
        self._versions.append((timestamp, version_dir))
        self.prune()
        return version_dir

    def latest_version(self) -> Tuple[str, Path]:
        if not self._versions:
            raise RuntimeError("No dataset versions available")
        return self._versions[-1]

    def split(self, data: List[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
        total = len(data)
        train_end = int(total * self._config.train_split)
        val_end = train_end + int(total * self._config.val_split)
        return {
            "train": data[:train_end],
            "val": data[train_end:val_end],
            "test": data[val_end:],
        }

    def metadata(self) -> List[Dict[str, object]]:
        return [
            {"version": version, "path": str(path)}
            for version, path in self._versions
        ]

    def prune(self) -> None:
        while len(self._versions) > self._config.retention:
            _, path = self._versions.popleft()
            self._delete_version(path)

    def _delete_version(self, path: Path) -> None:
        shutil.rmtree(path, ignore_errors=True)

    def export_manifest(self, destination: Path) -> None:
        manifest = {
            "created": datetime.utcnow().isoformat(),
            "versions": self.metadata(),
            "config": asdict(self._config),
        }
        destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
