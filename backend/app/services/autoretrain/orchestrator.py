"""End-to-end orchestration for automated retraining."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Sequence

from .ab_testing import ABTestManager
from .annotation import AnnotationPipeline
from .config import AutoRetrainConfig
from .data_collection import DataCollector
from .dataset import DatasetRepository
from .entities import AnnotatedSample
from .evaluation import EvaluationSuite
from .hparam import HyperParameterSearch
from .mlflow_integration import MLFlowTracker
from .monitoring import MonitoringDashboard
from .rollback import RollbackManager
from .training import TrainingOrchestrator


class AutoRetrainOrchestrator:
    """Coordinates collection, training, evaluation, and logging."""

    def __init__(self, config: AutoRetrainConfig) -> None:
        config.validate()
        self._config = config
        self._collector = DataCollector(config.data)
        self._annotator = AnnotationPipeline(config.annotation)
        self._dataset = DatasetRepository(config.dataset)
        self._search = HyperParameterSearch(config.hyperparameters)
        self._evaluator = EvaluationSuite(config.evaluation)
        self._trainer = TrainingOrchestrator(config.training, self._search)
        self._abtest = ABTestManager(config.abtest)
        self._rollback = RollbackManager(config.rollback)
        self._tracker = MLFlowTracker(config.mlflow)
        self._monitor = MonitoringDashboard(config.monitoring)

    def run_cycle(self) -> Dict[str, object]:
        raw_samples = self._collector.collect_batch()
        annotated = self._annotator.annotate(raw_samples)
        dataset_path = self._dataset.create_version(annotated)
        split = self._dataset.split(annotated)

        artifact, training_metrics = self._trainer.train(split, self._evaluator)
        evaluation_set: Sequence[AnnotatedSample] = split.test or split.val or split.train
        metrics = self._evaluator.evaluate(evaluation_set, artifact)
        validation = self._evaluator.validate(metrics)

        run_dir = self._tracker.start_run(artifact.params)
        self._tracker.log_metrics(run_dir, metrics)
        self._tracker.log_artifact(run_dir, "validation", validation)

        self._rollback.register(artifact.version, metrics)
        for name, value in metrics.items():
            self._monitor.record(name, value)

        return {
            "dataset_path": str(dataset_path),
            "artifact_version": artifact.version,
            "metrics": metrics,
            "training_metrics": training_metrics,
            "validation": validation,
            "run_dir": str(run_dir),
        }

    def evaluate_deployment(self, control: Sequence[float], treatment: Sequence[float]) -> Dict[str, object]:
        experiment = self._abtest.run_experiment(
            experiment_id=f"exp-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            control=control,
            treatment=treatment,
        )
        return {
            "experiment_id": experiment.experiment_id,
            "p_value": experiment.p_value,
            "control_mean": experiment.control_mean,
            "treatment_mean": experiment.treatment_mean,
            "significant": experiment.significant,
        }

    def monitoring_snapshot(self) -> Dict[str, List[float]]:
        return self._monitor.snapshot()

    def best_model(self, metric: str) -> Dict[str, object] | None:
        candidate = self._rollback.best_candidate(metric)
        if not candidate:
            return None
        version, metrics = candidate
        return {"version": version, "metrics": metrics}
