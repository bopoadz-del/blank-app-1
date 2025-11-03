"""Lightweight MLflow-style tracker used for debugging the pipeline."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping

from .config import MLFlowConfig


class MLFlowTracker:
    """Persist parameters, metrics and artifacts to the filesystem."""

    def __init__(self, config: MLFlowConfig) -> None:
        self._config = config
        self._root = config.tracking_dir / config.experiment_name
        self._root.mkdir(parents=True, exist_ok=True)

    def start_run(self, params: Mapping[str, float]) -> Path:
        run_id = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        run_dir = self._root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(run_dir / "params.json", params)
        return run_dir

    def log_metrics(self, run_dir: Path, metrics: Mapping[str, float]) -> None:
        metrics_path = run_dir / "metrics.json"
        self._write_json(metrics_path, metrics)

    def log_artifact(self, run_dir: Path, name: str, payload: Mapping[str, float | bool]) -> Path:
        artifact_path = run_dir / f"{name}.json"
        self._write_json(artifact_path, payload)
        return artifact_path

    def _write_json(self, path: Path, payload: Mapping[str, object]) -> None:
        path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True))
