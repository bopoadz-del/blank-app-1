"""Configuration dataclasses for the automated retraining pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence


@dataclass(slots=True)
class DataCollectionConfig:
    """Configuration for gathering raw samples from multiple sources."""

    sources: Mapping[str, Iterable[Mapping[str, object]]]
    shuffle: bool = True
    limit: Optional[int] = None

    def validate(self) -> None:
        if not self.sources:
            raise ValueError("At least one data source must be provided")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive when provided")


@dataclass(slots=True)
class AnnotationConfig:
    """Controls consensus and auto-labelling heuristics."""

    labelers: Dict[str, float]
    consensus_threshold: float = 0.6
    auto_label_rules: Dict[str, float] = field(default_factory=dict)
    review_sample_rate: float = 0.1

    def validate(self) -> None:
        if not self.labelers:
            raise ValueError("labelers cannot be empty")
        if not 0 < self.consensus_threshold <= 1:
            raise ValueError("consensus_threshold must be within (0, 1]")
        if not 0 <= self.review_sample_rate <= 1:
            raise ValueError("review_sample_rate must be within [0, 1]")


@dataclass(slots=True)
class DatasetConfig:
    """Dataset storage and splitting behaviour."""

    root_dir: Path
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    retention: int = 3

    def validate(self) -> None:
        if self.root_dir.name == "":
            raise ValueError("root_dir must point to a concrete directory")
        if self.retention <= 0:
            raise ValueError("retention must be positive")
        if not 0 < self.train_ratio < 1:
            raise ValueError("train_ratio must be within (0, 1)")
        if not 0 <= self.val_ratio < 1:
            raise ValueError("val_ratio must be within [0, 1)")
        if self.train_ratio + self.val_ratio >= 1:
            raise ValueError("train_ratio + val_ratio must be less than 1")


@dataclass(slots=True)
class TrainingConfig:
    """Hyperparameters for a single training run."""

    max_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.1
    l2_strength: float = 0.0
    tolerance: float = 1e-4

    def validate(self) -> None:
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.l2_strength < 0:
            raise ValueError("l2_strength cannot be negative")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")


@dataclass(slots=True)
class HyperParameterConfig:
    """Search space for the training orchestrator."""

    learning_rates: Sequence[float]
    l2_strengths: Sequence[float] = field(default_factory=lambda: [0.0])
    batch_sizes: Sequence[int] = field(default_factory=lambda: [32])
    max_trials: Optional[int] = None

    def validate(self) -> None:
        if not self.learning_rates:
            raise ValueError("learning_rates cannot be empty")
        if any(rate <= 0 for rate in self.learning_rates):
            raise ValueError("learning_rates must be positive")
        if any(strength < 0 for strength in self.l2_strengths):
            raise ValueError("l2_strengths cannot contain negative values")
        if any(size <= 0 for size in self.batch_sizes):
            raise ValueError("batch_sizes must be positive")
        if self.max_trials is not None and self.max_trials <= 0:
            raise ValueError("max_trials must be positive when provided")


@dataclass(slots=True)
class EvaluationConfig:
    """Metric configuration for evaluation and validation."""

    metrics: Sequence[str] = field(default_factory=lambda: ["accuracy", "precision", "recall", "f1"])
    positive_label: int = 1
    thresholds: Dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        valid_metrics = {"accuracy", "precision", "recall", "f1"}
        if not self.metrics:
            raise ValueError("At least one metric must be configured")
        for metric in self.metrics:
            if metric not in valid_metrics:
                raise ValueError(f"Unsupported metric: {metric}")
        for metric, value in self.thresholds.items():
            if metric not in valid_metrics:
                raise ValueError(f"Unsupported threshold metric: {metric}")
            if value < 0:
                raise ValueError("Threshold values must be non-negative")


@dataclass(slots=True)
class ABTestConfig:
    """Parameters governing A/B experimentation."""

    significance_level: float = 0.05
    minimum_total: int = 50

    def validate(self) -> None:
        if not 0 < self.significance_level < 0.5:
            raise ValueError("significance_level must be within (0, 0.5)")
        if self.minimum_total <= 0:
            raise ValueError("minimum_total must be positive")


@dataclass(slots=True)
class RollbackConfig:
    """Retention policy for previously trained models."""

    max_versions: int = 5

    def validate(self) -> None:
        if self.max_versions <= 0:
            raise ValueError("max_versions must be positive")


@dataclass(slots=True)
class MLFlowConfig:
    """Lightweight MLflow compatible logging options."""

    tracking_dir: Path
    experiment_name: str = "auto-retrain"

    def validate(self) -> None:
        if not self.tracking_dir:
            raise ValueError("tracking_dir must be provided")
        if not self.experiment_name:
            raise ValueError("experiment_name must be provided")


@dataclass(slots=True)
class MonitoringConfig:
    """Sliding window for dashboard style aggregations."""

    window: int = 10

    def validate(self) -> None:
        if self.window <= 0:
            raise ValueError("window must be positive")


@dataclass(slots=True)
class AutoRetrainConfig:
    """Aggregated configuration for the automated retraining service."""

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
