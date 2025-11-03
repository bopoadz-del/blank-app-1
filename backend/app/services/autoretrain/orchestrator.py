"""End-to-end orchestration for automated retraining."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable, List

from .ab_testing import ABTestManager
from .annotation import AnnotationPipeline
from .config import AutoRetrainConfig
from .data_collection import DataCollector
from .dataset import DatasetRepository
from .evaluation import EvaluationSuite
from .hparam import HyperParameterSearch
from .mlflow_integration import MLFlowTracker
from .monitoring import MonitoringDashboard
from .rollback import RollbackManager
from .training import TrainingOrchestrator


class AutoRetrainOrchestrator:
    """Coordinates all components of the automated retraining workflow."""

    def __init__(self, config: AutoRetrainConfig) -> None:
        config.validate()
        self._config = config
        self._collector = DataCollector(config.data)
        self._annotator = AnnotationPipeline(config.annotation)
        self._dataset = DatasetRepository(config.dataset)
        self._search = HyperParameterSearch(config.hyperparameters)
        self._trainer = TrainingOrchestrator(config.training, self._search)
        self._evaluator = EvaluationSuite(config.evaluation)
        self._abtest = ABTestManager(config.abtest)
        self._rollback = RollbackManager(config.rollback)
        self._tracker = MLFlowTracker(config.mlflow)
        self._monitor = MonitoringDashboard(config.monitoring)

    def run_cycle(self) -> Dict[str, object]:
        raw_batch = self._collector.collect_batch()
        annotated = self._annotator.annotate(raw_batch)
        dataset_path = self._dataset.create_version(annotated)
        split = self._dataset.split(annotated)
        artifact = self._trainer.train(split)
        eval_predictions = self._simulate_predictions(split["test"], artifact.params)
        metrics = self._evaluator.evaluate(eval_predictions)
        validation = self._evaluator.validate(metrics)
        run_dir = self._tracker.start_run(artifact.params)
        self._tracker.log_metrics(run_dir, metrics)
        self._tracker.log_artifact(run_dir, "validation", validation)
        self._rollback.register(artifact.version, metrics)
        for name, value in metrics.items():
            self._monitor.record(name, value)
        payload = {
            "dataset_path": str(dataset_path),
            "artifact_version": artifact.version,
            "metrics": metrics,
            "validation": validation,
            "run_dir": str(run_dir),
        }
        return payload

    def evaluate_deployment(self, control: List[float], treatment: List[float]) -> Dict[str, object]:
        experiment = self._abtest.run_experiment(
            experiment_id=f"exp-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            control=control,
            treatment=treatment,
        )
        return {
            "experiment_id": experiment.experiment_id,
            "p_value": experiment.p_value,
            "significant": experiment.significant,
        }

    def _simulate_predictions(self, dataset: Iterable[Dict[str, object]], params: Dict[str, object]) -> List[Dict[str, float]]:
        return [
            {
                "label": 1 if idx % 2 == 0 else 0,
                "prediction": 1 if idx % 3 == 0 else 0,
            }
            for idx, _ in enumerate(dataset)
        ]
