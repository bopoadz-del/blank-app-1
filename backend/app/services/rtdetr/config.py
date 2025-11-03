"""Configuration objects for the RT-DETR TensorRT deployment pipeline.

The configuration schema deliberately mirrors the numerous knobs exposed by
high-throughput detection deployments.  Each section is separated into nested
`dataclasses` to ease validation and to simplify debugging scenarios when
profiling throughput or accuracy regressions.

The module avoids optional dependencies by relying purely on stdlib typing
utilities.  Validation helpers provide runtime guards to keep the configuration
sane even when values are injected from dynamic orchestrators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import json
import logging

LOGGER = logging.getLogger(__name__)


def _ensure_positive(name: str, value: float) -> float:
    if value <= 0:
        raise ValueError(f"{name} must be positive, received {value!r}")
    return value


def _ensure_non_negative(name: str, value: float) -> float:
    if value < 0:
        raise ValueError(f"{name} must be >= 0, received {value!r}")
    return value


def _ensure_ratio(name: str, value: float) -> float:
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be within [0, 1], received {value!r}")
    return value


def _ensure_enum(name: str, value: str, options: Iterable[str]) -> str:
    if value not in options:
        allowed = ", ".join(sorted(options))
        raise ValueError(f"{name} must be one of {allowed}, received {value!r}")
    return value


def _validate_path(path: Path, *, must_exist: bool = True) -> Path:
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Path {path} not found")
    return path


@dataclass
class RTDETRDatasetConfig:
    """Dataset-centric configuration options.

    Attributes
    ----------
    root: Base directory containing training/validation assets. The directory is
        optional during deployment but enables debugging when sample assets are
        available.
    class_names: Human-readable class names in model order.
    image_size: Input resolution used during TensorRT compilation. The
        preprocessing pipeline automatically adjusts input images to this shape
        while preserving aspect ratios.
    calibration_images: Paths used for INT8 calibration. When omitted the
        TensorRT builder reuses the dataset root to enumerate assets.
    """

    root: Path = field(default_factory=lambda: Path("data/rtdetr"))
    class_names: List[str] = field(default_factory=lambda: [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "bus",
        "truck",
        "train",
        "fire hydrant",
        "stop sign",
        "parking meter",
    ])
    image_size: Tuple[int, int] = (640, 640)
    calibration_images: List[Path] = field(default_factory=list)

    def __post_init__(self) -> None:
        LOGGER.debug("Validating dataset config: %s", self)
        if self.root:
            _validate_path(Path(self.root).expanduser())
        if not self.class_names:
            raise ValueError("class_names must not be empty")
        if len(self.image_size) != 2:
            raise ValueError("image_size must contain two dimensions")
        for value in self.image_size:
            _ensure_positive("image dimension", value)
        for path in self.calibration_images:
            _validate_path(Path(path), must_exist=False)

    def to_mapping(self) -> Dict[str, object]:
        return {
            "root": str(self.root),
            "class_names": list(self.class_names),
            "image_size": list(self.image_size),
            "calibration_images": [str(p) for p in self.calibration_images],
        }


@dataclass
class RTDETRModelConfig:
    """Model hyperparameters used by the architecture scaffold."""

    backbone: str = "rtdetr_resnet50"
    num_queries: int = 300
    num_classes: int = 80
    width_mult: float = 1.0
    depth_mult: float = 1.0
    use_dcn: bool = True
    layer_norm_eps: float = 1e-5
    positional_encoding_type: str = "sine"
    activation: str = "gelu"
    checkpoint: Optional[Path] = None

    def __post_init__(self) -> None:
        LOGGER.debug("Validating model config: %s", self)
        _ensure_positive("num_queries", self.num_queries)
        _ensure_positive("num_classes", self.num_classes)
        _ensure_positive("width_mult", self.width_mult)
        _ensure_positive("depth_mult", self.depth_mult)
        _ensure_positive("layer_norm_eps", self.layer_norm_eps)
        _ensure_enum(
            "positional_encoding_type",
            self.positional_encoding_type,
            {"sine", "learned"},
        )
        _ensure_enum(
            "activation",
            self.activation,
            {"relu", "gelu", "silu", "swish"},
        )
        if self.checkpoint:
            _validate_path(Path(self.checkpoint).expanduser(), must_exist=False)

    def to_mapping(self) -> Dict[str, object]:
        return {
            "backbone": self.backbone,
            "num_queries": self.num_queries,
            "num_classes": self.num_classes,
            "width_mult": self.width_mult,
            "depth_mult": self.depth_mult,
            "use_dcn": self.use_dcn,
            "layer_norm_eps": self.layer_norm_eps,
            "positional_encoding_type": self.positional_encoding_type,
            "activation": self.activation,
            "checkpoint": str(self.checkpoint) if self.checkpoint else None,
        }


@dataclass
class TensorRTPrecisionConfig:
    precision: str = "int8"
    calibrator_cache: Path = field(
        default_factory=lambda: Path("data/rtdetr/calibration.cache")
    )
    strict_types: bool = True
    workspace_size_bytes: int = 2 * 1024 * 1024 * 1024  # 2GB
    fallback_to_fp16: bool = True
    fallback_to_fp32: bool = False
    int8_opt_level: int = 3

    def __post_init__(self) -> None:
        LOGGER.debug("Validating precision config: %s", self)
        _ensure_enum("precision", self.precision, {"int8", "fp16", "fp32"})
        _ensure_enum("int8_opt_level", self.int8_opt_level, {0, 1, 2, 3})
        _validate_path(self.calibrator_cache, must_exist=False)
        if self.precision == "int8" and not self.strict_types:
            LOGGER.warning(
                "INT8 precision typically requires strict_types=True for "
                "deterministic performance."
            )

    def to_mapping(self) -> Dict[str, object]:
        return {
            "precision": self.precision,
            "calibrator_cache": str(self.calibrator_cache),
            "strict_types": self.strict_types,
            "workspace_size_bytes": self.workspace_size_bytes,
            "fallback_to_fp16": self.fallback_to_fp16,
            "fallback_to_fp32": self.fallback_to_fp32,
            "int8_opt_level": self.int8_opt_level,
        }


@dataclass
class TensorRTBuilderConfig:
    max_batch_size: int = 4
    dla_core: Optional[int] = None
    enable_refit: bool = False
    enable_sparsity: bool = True
    enable_tactic_sources: Tuple[str, ...] = ("cublas", "cudnn")
    profiling_verbosity: str = "detailed"

    def __post_init__(self) -> None:
        LOGGER.debug("Validating builder config: %s", self)
        _ensure_positive("max_batch_size", self.max_batch_size)
        if self.dla_core is not None and self.dla_core < 0:
            raise ValueError("dla_core must be >= 0 when provided")
        _ensure_enum(
            "profiling_verbosity",
            self.profiling_verbosity,
            {"layer_names_only", "detailed", "none"},
        )
        if not self.enable_tactic_sources:
            raise ValueError("enable_tactic_sources must not be empty")

    def to_mapping(self) -> Dict[str, object]:
        return {
            "max_batch_size": self.max_batch_size,
            "dla_core": self.dla_core,
            "enable_refit": self.enable_refit,
            "enable_sparsity": self.enable_sparsity,
            "enable_tactic_sources": list(self.enable_tactic_sources),
            "profiling_verbosity": self.profiling_verbosity,
        }


@dataclass
class PreprocessingConfig:
    letterbox: bool = True
    normalize: bool = True
    color_format: str = "bgr"
    stride: int = 32
    pad_value: float = 114.0
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    dataset_shape: Tuple[int, int] = (640, 640)
    max_queue_size: int = 8
    pinned_memory: bool = True
    asynchronous: bool = True

    def __post_init__(self) -> None:
        LOGGER.debug("Validating preprocessing config: %s", self)
        _ensure_positive("stride", self.stride)
        _ensure_non_negative("pad_value", self.pad_value)
        _ensure_enum("color_format", self.color_format, {"bgr", "rgb"})
        _ensure_positive("max_queue_size", self.max_queue_size)
        if len(self.dataset_shape) != 2:
            raise ValueError("dataset_shape must contain two dimensions")
        for value in self.dataset_shape:
            _ensure_positive("dataset_shape dimension", value)

    def to_mapping(self) -> Dict[str, object]:
        return {
            "letterbox": self.letterbox,
            "normalize": self.normalize,
            "color_format": self.color_format,
            "stride": self.stride,
            "pad_value": self.pad_value,
            "mean": list(self.mean),
            "std": list(self.std),
            "dataset_shape": list(self.dataset_shape),
            "max_queue_size": self.max_queue_size,
            "pinned_memory": self.pinned_memory,
            "asynchronous": self.asynchronous,
        }


@dataclass
class PostprocessingConfig:
    conf_threshold: float = 0.25
    nms_threshold: float = 0.5
    topk: int = 300
    max_detections: int = 200
    use_fast_nms: bool = True
    box_format: str = "cxcywh"

    def __post_init__(self) -> None:
        LOGGER.debug("Validating postprocessing config: %s", self)
        _ensure_ratio("conf_threshold", self.conf_threshold)
        _ensure_ratio("nms_threshold", self.nms_threshold)
        _ensure_positive("topk", self.topk)
        _ensure_positive("max_detections", self.max_detections)
        _ensure_enum("box_format", self.box_format, {"cxcywh", "xywh", "xyxy"})

    def to_mapping(self) -> Dict[str, object]:
        return {
            "conf_threshold": self.conf_threshold,
            "nms_threshold": self.nms_threshold,
            "topk": self.topk,
            "max_detections": self.max_detections,
            "use_fast_nms": self.use_fast_nms,
            "box_format": self.box_format,
        }


@dataclass
class BatchConfig:
    enable_dynamic_batching: bool = True
    max_batch_size: int = 16
    preferred_batch_sizes: Tuple[int, ...] = (1, 2, 4, 8, 16)
    capture_cuda_graphs: bool = True
    warmup_iterations: int = 16

    def __post_init__(self) -> None:
        LOGGER.debug("Validating batch config: %s", self)
        if self.enable_dynamic_batching:
            _ensure_positive("max_batch_size", self.max_batch_size)
            if any(size <= 0 for size in self.preferred_batch_sizes):
                raise ValueError("preferred_batch_sizes must contain positive values")
        _ensure_positive("warmup_iterations", self.warmup_iterations)

    def to_mapping(self) -> Dict[str, object]:
        return {
            "enable_dynamic_batching": self.enable_dynamic_batching,
            "max_batch_size": self.max_batch_size,
            "preferred_batch_sizes": list(self.preferred_batch_sizes),
            "capture_cuda_graphs": self.capture_cuda_graphs,
            "warmup_iterations": self.warmup_iterations,
        }


@dataclass
class MultiStreamConfig:
    stream_count: int = 4
    stream_binding_strategy: str = "round_robin"
    event_timing: bool = True
    overlap_transfers: bool = True

    def __post_init__(self) -> None:
        LOGGER.debug("Validating multi-stream config: %s", self)
        _ensure_positive("stream_count", self.stream_count)
        _ensure_enum(
            "stream_binding_strategy",
            self.stream_binding_strategy,
            {"round_robin", "least_loaded", "pinned"},
        )

    def to_mapping(self) -> Dict[str, object]:
        return {
            "stream_count": self.stream_count,
            "stream_binding_strategy": self.stream_binding_strategy,
            "event_timing": self.event_timing,
            "overlap_transfers": self.overlap_transfers,
        }


@dataclass
class ProfilingConfig:
    enable_profiling: bool = True
    capture_timeline: bool = True
    timeline_path: Path = field(
        default_factory=lambda: Path("data/rtdetr/profile/timeline.json")
    )
    aggregate_metrics: bool = True
    min_report_interval_s: float = 2.0

    def __post_init__(self) -> None:
        LOGGER.debug("Validating profiling config: %s", self)
        if self.enable_profiling:
            _validate_path(self.timeline_path.parent, must_exist=False)
            _ensure_positive("min_report_interval_s", self.min_report_interval_s)

    def to_mapping(self) -> Dict[str, object]:
        return {
            "enable_profiling": self.enable_profiling,
            "capture_timeline": self.capture_timeline,
            "timeline_path": str(self.timeline_path),
            "aggregate_metrics": self.aggregate_metrics,
            "min_report_interval_s": self.min_report_interval_s,
        }


@dataclass
class DeploymentConfig:
    engine_cache_dir: Path = field(default_factory=lambda: Path("data/rtdetr/engines"))
    engine_filename: str = "rtdetr_int8.plan"
    hot_reload: bool = True
    watchdog_interval_s: float = 30.0
    autorebuild_on_schema_change: bool = True
    export_onnx: bool = True
    onnx_path: Path = field(default_factory=lambda: Path("data/rtdetr/model.onnx"))

    def __post_init__(self) -> None:
        LOGGER.debug("Validating deployment config: %s", self)
        _validate_path(self.engine_cache_dir, must_exist=False)
        if self.watchdog_interval_s <= 0:
            raise ValueError("watchdog_interval_s must be positive")
        if self.export_onnx:
            _validate_path(self.onnx_path.parent, must_exist=False)

    def to_mapping(self) -> Dict[str, object]:
        return {
            "engine_cache_dir": str(self.engine_cache_dir),
            "engine_filename": self.engine_filename,
            "hot_reload": self.hot_reload,
            "watchdog_interval_s": self.watchdog_interval_s,
            "autorebuild_on_schema_change": self.autorebuild_on_schema_change,
            "export_onnx": self.export_onnx,
            "onnx_path": str(self.onnx_path),
        }


@dataclass
class ErrorHandlingConfig:
    retry_attempts: int = 3
    retry_backoff_s: float = 1.5
    fallback_precision: Optional[str] = "fp16"
    capture_stack_traces: bool = True
    verbose_logging: bool = True

    def __post_init__(self) -> None:
        LOGGER.debug("Validating error handling config: %s", self)
        _ensure_non_negative("retry_attempts", self.retry_attempts)
        _ensure_non_negative("retry_backoff_s", self.retry_backoff_s)
        if self.fallback_precision:
            _ensure_enum(
                "fallback_precision",
                self.fallback_precision,
                {"int8", "fp16", "fp32"},
            )

    def to_mapping(self) -> Dict[str, object]:
        return {
            "retry_attempts": self.retry_attempts,
            "retry_backoff_s": self.retry_backoff_s,
            "fallback_precision": self.fallback_precision,
            "capture_stack_traces": self.capture_stack_traces,
            "verbose_logging": self.verbose_logging,
        }


@dataclass
class RTDETRConfig:
    dataset: RTDETRDatasetConfig = field(default_factory=RTDETRDatasetConfig)
    model: RTDETRModelConfig = field(default_factory=RTDETRModelConfig)
    precision: TensorRTPrecisionConfig = field(default_factory=TensorRTPrecisionConfig)
    builder: TensorRTBuilderConfig = field(default_factory=TensorRTBuilderConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)
    batching: BatchConfig = field(default_factory=BatchConfig)
    multistream: MultiStreamConfig = field(default_factory=MultiStreamConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    error_handling: ErrorHandlingConfig = field(default_factory=ErrorHandlingConfig)

    extra_metadata: MutableMapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        LOGGER.debug("Instantiated RT-DETR config: %s", self)

    def to_mapping(self) -> Dict[str, object]:
        return {
            "dataset": self.dataset.to_mapping(),
            "model": self.model.to_mapping(),
            "precision": self.precision.to_mapping(),
            "builder": self.builder.to_mapping(),
            "preprocessing": self.preprocessing.to_mapping(),
            "postprocessing": self.postprocessing.to_mapping(),
            "batching": self.batching.to_mapping(),
            "multistream": self.multistream.to_mapping(),
            "profiling": self.profiling.to_mapping(),
            "deployment": self.deployment.to_mapping(),
            "error_handling": self.error_handling.to_mapping(),
            "extra_metadata": dict(self.extra_metadata),
        }

    def update_from_mapping(self, mapping: Mapping[str, object]) -> None:
        LOGGER.debug("Updating RT-DETR config from mapping: %s", mapping)
        for key, value in mapping.items():
            if not hasattr(self, key):
                LOGGER.warning("Ignoring unknown config key: %s", key)
                continue
            attr = getattr(self, key)
            if hasattr(attr, "__dict__") and isinstance(value, Mapping):
                for nested_key, nested_value in value.items():
                    if hasattr(attr, nested_key):
                        setattr(attr, nested_key, nested_value)
            else:
                setattr(self, key, value)

    @classmethod
    def from_json(cls, path: Path) -> "RTDETRConfig":
        LOGGER.info("Loading RT-DETR configuration from %s", path)
        content = path.read_text()
        data = json.loads(content)
        config = cls()
        config.update_from_mapping(data)
        return config

    def dump_json(self, path: Path) -> None:
        LOGGER.info("Writing RT-DETR configuration to %s", path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_mapping(), indent=2, sort_keys=True))


def merge_configs(base: RTDETRConfig, overrides: Optional[Mapping[str, object]]) -> RTDETRConfig:
    if not overrides:
        return base
    LOGGER.debug("Merging RT-DETR configs with overrides: %s", overrides)
    merged = RTDETRConfig()
    merged.update_from_mapping(base.to_mapping())
    merged.update_from_mapping(overrides)
    return merged
