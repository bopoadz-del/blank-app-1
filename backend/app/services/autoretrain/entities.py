"""Shared dataclasses used across the automated retraining pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(slots=True)
class RawSample:
    """A single unlabelled or weakly labelled training sample."""

    features: List[float]
    label: Optional[int] = None
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class AnnotatedSample:
    """A fully annotated sample ready for training."""

    features: List[float]
    label: int
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class DatasetSplit:
    """A dataset split into train/validation/test partitions."""

    train: List[AnnotatedSample]
    val: List[AnnotatedSample]
    test: List[AnnotatedSample]


@dataclass(slots=True)
class TrainingArtifact:
    """Serializable representation of a trained logistic regression model."""

    version: str
    params: Dict[str, float]
    weights: List[float]
    bias: float
