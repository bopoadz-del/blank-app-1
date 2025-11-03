"""Configuration schemas for the automated retraining pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


@dataclass(slots=True)
class DataCollectionConfig:
    """Configuration for data collection sources and batching."""

    sources: Dict[str, str]
    batch_size: int = 256
    poll_interval: timedelta = timedelta(minutes=5)
    max_latency: timedelta = timedelta(hours=1)
    cache_dir: Path = Path("/tmp/auto_retrain/cache")

    def validate(self) -> None:
        if not self.sources:
            raise ValueError("At least one data source must be configured")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.poll_interval <= timedelta(0):
            raise ValueError("Poll interval must be positive")
        if self.max_latency < self.poll_interval:
            raise ValueError("Max latency must exceed poll interval")


@dataclass(slots=True)
class AnnotationConfig:
    """Configuration for annotation pipelines and labelers."""

    labelers: Dict[str, float]
    consensus_threshold: float = 0.7
    auto_label_rules: Dict[str, Dict[str, float]] = field(default_factory=dict)
    review_sample_rate: float = 0.1

    def validate(self) -> None:
        if not self.labelers:
            raise ValueError("Annotation requires at least one labeler")
        if not 0 < self.consensus_threshold <= 1:
            raise ValueError("Consensus threshold must be in (0, 1]")
        if not 0 <= self.review_sample_rate <= 1:
            raise ValueError("Review sample rate must be within [0, 1]")


@dataclass(slots=True)
class DatasetConfig:
    """Configuration for dataset versioning and storage."""

    root_dir: Path
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    retention: int = 5

    def validate(self) -> None:
        total = self.train_split + self.val_split + self.test_split
        if abs(total - 1.0) > 1e-6:
            raise ValueError("Dataset splits must sum to 1.0")
        if self.retention <= 0:
            raise ValueError("Retention must be positive")


@dataclass(slots=True)
class TrainingConfig:
    """Configuration for training orchestration."""

    max_epochs: int = 50
    early_stopping_patience: int = 5
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    checkpoint_dir: Path = Path("/tmp/auto_retrain/checkpoints")

    def validate(self) -> None:
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience cannot be negative")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")


@dataclass(slots=True)
class HyperParameterConfig:
    """Configuration for hyperparameter search spaces."""

    search_space: Dict[str, Sequence]
    max_trials: int = 20
    parallel_trials: int = 1

    def validate(self) -> None:
        if not self.search_space:
            raise ValueError("search_space must not be empty")
        if self.max_trials <= 0:
            raise ValueError("max_trials must be positive")
        if self.parallel_trials <= 0:
            raise ValueError("parallel_trials must be positive")


@dataclass(slots=True)
class EvaluationConfig:
    """Configuration for evaluation metrics and thresholds."""

    metrics: Iterable[str]
    thresholds: Dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        if not list(self.metrics):
            raise ValueError("At least one evaluation metric must be provided")
        for name, value in self.thresholds.items():
            if value < 0:
                raise ValueError(f"Threshold for {name} must be non-negative")


@dataclass(slots=True)
class ABTestConfig:
    """Configuration for A/B testing experiments."""

    experiment_window: timedelta = timedelta(days=7)
    min_users: int = 1000
    significance_level: float = 0.05

    def validate(self) -> None:
        if self.min_users <= 0:
            raise ValueError("min_users must be positive")
        if not 0 < self.significance_level < 0.5:
            raise ValueError("significance_level must be in (0, 0.5)")


@dataclass(slots=True)
class RollbackConfig:
    """Configuration for rollback policies."""

    max_versions: int = 10
    safety_checks: List[str] = field(default_factory=lambda: ["performance", "latency"])

    def validate(self) -> None:
        if self.max_versions <= 0:
            raise ValueError("max_versions must be positive")


@dataclass(slots=True)
class MLFlowConfig:
    """Configuration for MLflow-compatible logging."""

    tracking_uri: str
    experiment_name: str
    run_name_template: str = "auto-retrain-{timestamp}"
    artifact_location: Optional[str] = None

    def validate(self) -> None:
        if not self.tracking_uri:
            raise ValueError("tracking_uri must be provided")
        if not self.experiment_name:
            raise ValueError("experiment_name must be provided")


@dataclass(slots=True)
class MonitoringConfig:
    """Configuration for monitoring dashboards and alerts."""

    refresh_interval: timedelta = timedelta(minutes=1)
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    retention_period: timedelta = timedelta(days=30)

    def validate(self) -> None:
        if self.refresh_interval <= timedelta(0):
            raise ValueError("refresh_interval must be positive")
        if self.retention_period <= timedelta(0):
            raise ValueError("retention_period must be positive")


@dataclass(slots=True)
class AutoRetrainConfig:
    """Aggregate configuration for automated retraining."""

    data: DataCollectionConfig
    annotation: AnnotationConfig
    dataset: DatasetConfig
    training: TrainingConfig
    hyperparameters: HyperParameterConfig
    evaluation: EvaluationConfig
    abtest: ABTestConfig
    rollback: RollbackConfig
    mlflow: MLFlowConfig
    monitoring: MonitoringConfig

    def validate(self) -> None:
        self.data.validate()
        self.annotation.validate()
        self.dataset.validate()
        self.training.validate()
        self.hyperparameters.validate()
        self.evaluation.validate()
        self.abtest.validate()
        self.rollback.validate()
        self.mlflow.validate()
        self.monitoring.validate()
