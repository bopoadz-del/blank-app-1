<<<<<codex/debug-application-issues-vqhnb4
"""Lightweight MLflow-style tracker used for debugging the pipeline."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping
=======
"""Lightweight MLflow-compatible tracking."""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
>>>>> main

from .config import MLFlowConfig


class MLFlowTracker:
<<<<< codex/debug-application-issues-vqhnb4
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
=======
    """Persists run metadata in an MLflow-compatible directory layout."""

    def __init__(self, config: MLFlowConfig) -> None:
        config.validate()
        self._config = config
        self._base_path = Path(config.artifact_location or "./mlruns")
        self._base_path.mkdir(parents=True, exist_ok=True)

    def start_run(self, params: Dict[str, object]) -> Path:
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        run_name = self._config.run_name_template.format(timestamp=timestamp)
        run_dir = self._base_path / self._config.experiment_name / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "tracking_uri": self._config.tracking_uri,
            "run_name": run_name,
            "params": params,
            "started_at": datetime.utcnow().isoformat(),
        }
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return run_dir

    def log_metrics(self, run_dir: Path, metrics: Dict[str, float]) -> None:
        payload = {
            "logged_at": datetime.utcnow().isoformat(),
            "metrics": metrics,
        }
        (run_dir / "metrics.jsonl").write_text(json.dumps(payload), encoding="utf-8")

    def log_artifact(self, run_dir: Path, name: str, content: Dict[str, object]) -> None:
        path = run_dir / f"{name}.json"
        path.write_text(json.dumps(content, indent=2), encoding="utf-8")

    def export_config(self, destination: Path) -> None:
        destination.write_text(json.dumps(asdict(self._config), indent=2), encoding="utf-8")
>>>>> main
