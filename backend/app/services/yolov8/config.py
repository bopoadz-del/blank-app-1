"""Configuration schema for the YOLOv8 TensorRT deployment stack.

The :mod:`config` module centralises all runtime options required to drive the
end-to-end inference and training pipeline.  A single configuration object is
passed across the preprocessing, engine build, execution, tracking and
monitoring layers, which makes the behaviour of the system reproducible and
simple to reason about during debugging sessions.  Each configuration block is
modelled as a dataclass with explicit validation helpers so that misconfigured
runs can be flagged early before GPU resources are touched.

The settings exposed here are intentionally verbose – we surface everything that
an advanced deployment would typically need in production such as calibrator
parameters, CUDA stream tuning flags, batching preferences, and hooks for custom
plugins.  The resulting module is long but having a canonical, well-documented
configuration schema dramatically improves maintainability when the pipeline is
shared between research and operations teams.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple


@dataclass(frozen=True)
class DatasetPaths:
    """Filesystem locations required during preprocessing and quantisation.

    Attributes
    ----------
    images: List[Path]
        Input image directories used for calibration and benchmarking.  Every
        directory will be scanned recursively for supported image formats.
    annotations: Optional[Path]
        Optional COCO-style annotation file.  If provided the calibration
        routines can compute per-class statistics which leads to better INT8
        scales.
    cache_dir: Path
        Location where intermediate artefacts such as resized calibration
        batches and profiling traces are stored.  The directory will be created
        automatically when missing.
    """

    images: List[Path]
    annotations: Optional[Path]
    cache_dir: Path

    def ensure_exists(self) -> None:
        """Create the cache directory when it does not already exist."""

        self.cache_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class TensorRTBuildConfig:
    """Options that control how the TensorRT engine is constructed.

    These options cover the network definition, optimisation profile, precision
    selection, and plugin registration.  The defaults target a fast YOLOv8 nano
    deployment that prioritises throughput without sacrificing detection
    quality.  More advanced deployments can override the flags in production
    configuration files.
    """

    max_workspace_size: int = 2 << 30
    fp16: bool = True
    int8: bool = True
    strict_types: bool = False
    max_batch_size: int = 8
    calibration_batch_size: int = 16
    calibration_batches: int = 128
    dynamic_shapes: bool = True
    plugin_library: Optional[Path] = None
    engine_cache: Optional[Path] = None
    enable_dla: bool = False
    dla_core: int = 0
    builder_optimization_level: int = 5
    profiling_verbosity: str = "DETAILED"

    def optimisation_profiles(self) -> List[Tuple[int, int, int]]:
        """Return the default dynamic shape profile for (min, opt, max) sizes."""

        min_batch = max(1, self.max_batch_size // 8)
        opt_batch = max(1, self.max_batch_size // 4)
        max_batch = self.max_batch_size
        return [(min_batch, opt_batch, max_batch)]


@dataclass(frozen=True)
class QuantizationConfig:
    """Fine-grained settings for INT8 calibration and post-training tweaking."""

    calibrator_cache: Path
    per_channel: bool = True
    percentile: Optional[float] = 99.5
    entropy: bool = False
    load_existing: bool = True
    dynamic_range_override: Optional[Mapping[str, Tuple[float, float]]] = None

    def should_use_percentile(self) -> bool:
        return self.percentile is not None and not self.entropy


@dataclass(frozen=True)
class PreprocessingConfig:
    """Parameters that drive letterbox resizing and pixel normalisation."""

    input_width: int
    input_height: int
    stride: int = 32
    padding_color: Tuple[int, int, int] = (114, 114, 114)
    normalize: bool = True
    mean: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    max_batch: int = 16
    cuda_streams: int = 2
    use_pinned_memory: bool = True


@dataclass(frozen=True)
class PostprocessingConfig:
    """Settings used by the decode and Non-Maximum Suppression steps."""

    score_threshold: float = 0.25
    nms_iou_threshold: float = 0.7
    max_detections: int = 300
    max_per_class: int = 100
    multi_label: bool = False
    use_fast_nms: bool = True
    pre_nms_topk: int = 1024
    class_agnostic: bool = False


@dataclass(frozen=True)
class TrackerConfig:
    """Configuration for the integrated multi-object tracker."""

    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3
    embedding_similarity_threshold: float = 0.5
    motion_lambda: float = 0.2
    max_velocity: float = 20.0
    new_track_score_threshold: float = 0.4
    reid_history: int = 30
    matching_strategy: str = "iou"


@dataclass(frozen=True)
class ProfilingConfig:
    """Parameters that drive GPU timing and system profiling."""

    enable: bool = True
    warmup_runs: int = 10
    measurement_runs: int = 100
    export_trace: bool = True
    export_path: Optional[Path] = None
    sync_streams: bool = True


@dataclass(frozen=True)
class DeploymentPaths:
    """Filesystem paths used for engines, calibration data and assets."""

    engine: Path
    plan: Optional[Path]
    calibration_cache: Path
    profile_output: Optional[Path]


@dataclass(frozen=True)
class DebugConfig:
    """Flags that control verbose logging and debug rendering options."""

    verbose: bool = False
    dump_tensors: bool = False
    dump_dir: Optional[Path] = None
    visualize_preprocessing: bool = False
    visualize_postprocessing: bool = False
    visualize_tracking: bool = False
    enable_profiler: bool = True


@dataclass(frozen=True)
class YOLOv8Config:
    """Top-level configuration object covering every pipeline component."""

    dataset: DatasetPaths
    tensorrt: TensorRTBuildConfig
    quantization: QuantizationConfig
    preprocessing: PreprocessingConfig
    postprocessing: PostprocessingConfig
    tracker: TrackerConfig
    profiling: ProfilingConfig
    deployment: DeploymentPaths
    debug: DebugConfig = field(default_factory=DebugConfig)
    custom_properties: MutableMapping[str, str] = field(default_factory=dict)

    def summary(self) -> Dict[str, object]:
        """Return a serialisable summary for logging and UI display."""

        return {
            "dataset": {
                "images": [str(path) for path in self.dataset.images],
                "annotations": str(self.dataset.annotations)
                if self.dataset.annotations
                else None,
                "cache_dir": str(self.dataset.cache_dir),
            },
            "tensorrt": {
                "max_workspace_size": self.tensorrt.max_workspace_size,
                "fp16": self.tensorrt.fp16,
                "int8": self.tensorrt.int8,
                "max_batch_size": self.tensorrt.max_batch_size,
                "dynamic_shapes": self.tensorrt.dynamic_shapes,
                "builder_optimization_level": self.tensorrt.builder_optimization_level,
            },
            "quantization": {
                "per_channel": self.quantization.per_channel,
                "percentile": self.quantization.percentile,
                "entropy": self.quantization.entropy,
                "calibrator_cache": str(self.quantization.calibrator_cache),
            },
            "preprocessing": {
                "input_width": self.preprocessing.input_width,
                "input_height": self.preprocessing.input_height,
                "normalize": self.preprocessing.normalize,
                "cuda_streams": self.preprocessing.cuda_streams,
            },
            "postprocessing": {
                "score_threshold": self.postprocessing.score_threshold,
                "nms_iou_threshold": self.postprocessing.nms_iou_threshold,
                "max_detections": self.postprocessing.max_detections,
            },
            "tracker": {
                "max_age": self.tracker.max_age,
                "min_hits": self.tracker.min_hits,
                "matching_strategy": self.tracker.matching_strategy,
            },
            "profiling": {
                "enable": self.profiling.enable,
                "warmup_runs": self.profiling.warmup_runs,
                "measurement_runs": self.profiling.measurement_runs,
            },
            "deployment": {
                "engine": str(self.deployment.engine),
                "plan": str(self.deployment.plan) if self.deployment.plan else None,
                "calibration_cache": str(self.deployment.calibration_cache),
                "profile_output": str(self.deployment.profile_output)
                if self.deployment.profile_output
                else None,
            },
            "debug": {
                "verbose": self.debug.verbose,
                "dump_tensors": self.debug.dump_tensors,
                "visualize_preprocessing": self.debug.visualize_preprocessing,
                "visualize_postprocessing": self.debug.visualize_postprocessing,
                "visualize_tracking": self.debug.visualize_tracking,
            },
            "custom_properties": dict(self.custom_properties),
        }

    def with_overrides(self, **overrides: object) -> "YOLOv8Config":
        """Return a copy with selected attributes updated.

        The method accepts a flat dictionary where keys use dotted notation to
        address nested dataclass attributes (for example
        ``"preprocessing.input_width"``).  Values that are themselves mappings
        will be merged recursively.  This convenience helper is heavily used by
        the CLI tooling and automated deployment scripts where command line
        arguments override defaults derived from configuration files.
        """

        data: Dict[str, object] = self.summary()
        for key, value in overrides.items():
            self._apply_override(data, key.split("."), value)

        return YOLOv8Config(
            dataset=self.dataset,
            tensorrt=self.tensorrt,
            quantization=self.quantization,
            preprocessing=self.preprocessing,
            postprocessing=self.postprocessing,
            tracker=self.tracker,
            profiling=self.profiling,
            deployment=self.deployment,
            debug=self.debug,
            custom_properties=dict(self.custom_properties, **overrides.get("custom_properties", {})),
        )

    def _apply_override(self, mapping: MutableMapping[str, object], path: List[str], value: object) -> None:
        key = path[0]
        if len(path) == 1:
            mapping[key] = value
            return
        nested = mapping.setdefault(key, {})
        if not isinstance(nested, MutableMapping):
            raise TypeError(f"Cannot override non-mapping attribute '{key}'")
        self._apply_override(nested, path[1:], value)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "YOLOv8Config":
        """Construct a configuration object from a nested dictionary."""

        dataset_cfg = payload.get("dataset", {})
        tensorrt_cfg = payload.get("tensorrt", {})
        quant_cfg = payload.get("quantization", {})
        preprocess_cfg = payload.get("preprocessing", {})
        postprocess_cfg = payload.get("postprocessing", {})
        tracker_cfg = payload.get("tracker", {})
        profiling_cfg = payload.get("profiling", {})
        deployment_cfg = payload.get("deployment", {})
        debug_cfg = payload.get("debug", {})

        dataset = DatasetPaths(
            images=[Path(p) for p in dataset_cfg.get("images", [])],
            annotations=(
                Path(dataset_cfg["annotations"]) if dataset_cfg.get("annotations") else None
            ),
            cache_dir=Path(dataset_cfg.get("cache_dir", "./cache")),
        )

        tensorrt = TensorRTBuildConfig(
            max_workspace_size=int(tensorrt_cfg.get("max_workspace_size", 2 << 30)),
            fp16=bool(tensorrt_cfg.get("fp16", True)),
            int8=bool(tensorrt_cfg.get("int8", True)),
            strict_types=bool(tensorrt_cfg.get("strict_types", False)),
            max_batch_size=int(tensorrt_cfg.get("max_batch_size", 8)),
            calibration_batch_size=int(tensorrt_cfg.get("calibration_batch_size", 16)),
            calibration_batches=int(tensorrt_cfg.get("calibration_batches", 128)),
            dynamic_shapes=bool(tensorrt_cfg.get("dynamic_shapes", True)),
            plugin_library=(
                Path(tensorrt_cfg["plugin_library"]) if tensorrt_cfg.get("plugin_library") else None
            ),
            engine_cache=(
                Path(tensorrt_cfg["engine_cache"]) if tensorrt_cfg.get("engine_cache") else None
            ),
            enable_dla=bool(tensorrt_cfg.get("enable_dla", False)),
            dla_core=int(tensorrt_cfg.get("dla_core", 0)),
            builder_optimization_level=int(tensorrt_cfg.get("builder_optimization_level", 5)),
            profiling_verbosity=str(tensorrt_cfg.get("profiling_verbosity", "DETAILED")),
        )

        quantization = QuantizationConfig(
            calibrator_cache=Path(quant_cfg.get("calibrator_cache", "./calib.cache")),
            per_channel=bool(quant_cfg.get("per_channel", True)),
            percentile=(float(quant_cfg["percentile"]) if quant_cfg.get("percentile") else None),
            entropy=bool(quant_cfg.get("entropy", False)),
            load_existing=bool(quant_cfg.get("load_existing", True)),
            dynamic_range_override=quant_cfg.get("dynamic_range_override"),
        )

        preprocessing = PreprocessingConfig(
            input_width=int(preprocess_cfg.get("input_width", 640)),
            input_height=int(preprocess_cfg.get("input_height", 640)),
            stride=int(preprocess_cfg.get("stride", 32)),
            padding_color=tuple(preprocess_cfg.get("padding_color", (114, 114, 114))),
            normalize=bool(preprocess_cfg.get("normalize", True)),
            mean=tuple(preprocess_cfg.get("mean", (0.0, 0.0, 0.0))),
            std=tuple(preprocess_cfg.get("std", (1.0, 1.0, 1.0))),
            max_batch=int(preprocess_cfg.get("max_batch", 16)),
            cuda_streams=int(preprocess_cfg.get("cuda_streams", 2)),
            use_pinned_memory=bool(preprocess_cfg.get("use_pinned_memory", True)),
        )

        postprocessing = PostprocessingConfig(
            score_threshold=float(postprocess_cfg.get("score_threshold", 0.25)),
            nms_iou_threshold=float(postprocess_cfg.get("nms_iou_threshold", 0.7)),
            max_detections=int(postprocess_cfg.get("max_detections", 300)),
            max_per_class=int(postprocess_cfg.get("max_per_class", 100)),
            multi_label=bool(postprocess_cfg.get("multi_label", False)),
            use_fast_nms=bool(postprocess_cfg.get("use_fast_nms", True)),
            pre_nms_topk=int(postprocess_cfg.get("pre_nms_topk", 1024)),
            class_agnostic=bool(postprocess_cfg.get("class_agnostic", False)),
        )

        tracker = TrackerConfig(
            max_age=int(tracker_cfg.get("max_age", 30)),
            min_hits=int(tracker_cfg.get("min_hits", 3)),
            iou_threshold=float(tracker_cfg.get("iou_threshold", 0.3)),
            embedding_similarity_threshold=float(
                tracker_cfg.get("embedding_similarity_threshold", 0.5)
            ),
            motion_lambda=float(tracker_cfg.get("motion_lambda", 0.2)),
            max_velocity=float(tracker_cfg.get("max_velocity", 20.0)),
            new_track_score_threshold=float(tracker_cfg.get("new_track_score_threshold", 0.4)),
            reid_history=int(tracker_cfg.get("reid_history", 30)),
            matching_strategy=str(tracker_cfg.get("matching_strategy", "iou")),
        )

        profiling = ProfilingConfig(
            enable=bool(profiling_cfg.get("enable", True)),
            warmup_runs=int(profiling_cfg.get("warmup_runs", 10)),
            measurement_runs=int(profiling_cfg.get("measurement_runs", 100)),
            export_trace=bool(profiling_cfg.get("export_trace", True)),
            export_path=(
                Path(profiling_cfg["export_path"]) if profiling_cfg.get("export_path") else None
            ),
            sync_streams=bool(profiling_cfg.get("sync_streams", True)),
        )

        deployment = DeploymentPaths(
            engine=Path(deployment_cfg.get("engine", "./yolov8.plan")),
            plan=(Path(deployment_cfg["plan"]) if deployment_cfg.get("plan") else None),
            calibration_cache=Path(deployment_cfg.get("calibration_cache", "./calib.cache")),
            profile_output=(
                Path(deployment_cfg["profile_output"]) if deployment_cfg.get("profile_output") else None
            ),
        )

        debug = DebugConfig(
            verbose=bool(debug_cfg.get("verbose", False)),
            dump_tensors=bool(debug_cfg.get("dump_tensors", False)),
            dump_dir=(Path(debug_cfg["dump_dir"]) if debug_cfg.get("dump_dir") else None),
            visualize_preprocessing=bool(debug_cfg.get("visualize_preprocessing", False)),
            visualize_postprocessing=bool(debug_cfg.get("visualize_postprocessing", False)),
            visualize_tracking=bool(debug_cfg.get("visualize_tracking", False)),
            enable_profiler=bool(debug_cfg.get("enable_profiler", True)),
        )

        custom = dict(payload.get("custom_properties", {}))

        return cls(
            dataset=dataset,
            tensorrt=tensorrt,
            quantization=quantization,
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            tracker=tracker,
            profiling=profiling,
            deployment=deployment,
            debug=debug,
            custom_properties=custom,
        )

    def as_dict(self) -> Dict[str, object]:
        """Return the configuration as a deep dictionary."""

        return self.summary()

    def describe(self) -> str:
        """Return a multi-line human readable description of the config."""

        lines: List[str] = ["YOLOv8 TensorRT Configuration"]
        summary = self.summary()
        for section, values in summary.items():
            lines.append(f"[{section.upper()}]")
            if isinstance(values, Mapping):
                for key, value in values.items():
                    lines.append(f"{key:>24}: {value}")
            else:
                lines.append(str(values))
            lines.append("")
        return "\n".join(lines)

    def validate(self) -> List[str]:
        """Perform basic validation returning a list of warnings."""

        warnings: List[str] = []
        if not self.dataset.images:
            warnings.append("No dataset images configured; calibration may fail.")
        if self.tensorrt.int8 and not self.quantization.calibrator_cache.exists():
            warnings.append(
                "INT8 is enabled but the calibrator cache does not exist.  "
                "Calibration will take longer on the first run."
            )
        if self.preprocessing.input_width % self.preprocessing.stride != 0:
            warnings.append("Input width must be divisible by the network stride.")
        if self.preprocessing.input_height % self.preprocessing.stride != 0:
            warnings.append("Input height must be divisible by the network stride.")
        if self.postprocessing.score_threshold <= 0 or self.postprocessing.score_threshold >= 1:
            warnings.append("Score threshold should be within (0, 1).")
        if self.postprocessing.nms_iou_threshold <= 0 or self.postprocessing.nms_iou_threshold > 1:
            warnings.append("NMS IoU threshold should be within (0, 1].")
        if self.tracker.matching_strategy not in {"iou", "embedding", "hybrid"}:
            warnings.append("Unknown tracking matching strategy configured.")
        if self.profiling.export_trace and not self.profiling.export_path:
            warnings.append("Profiling trace export is enabled but no output path is set.")
        if self.debug.dump_tensors and not self.debug.dump_dir:
            warnings.append("Tensor dumping requested but dump directory is missing.")
        return warnings

    def iter_image_dirs(self) -> Iterable[Path]:
        """Iterate over unique image directories used for calibration."""

        seen = set()
        for image_dir in self.dataset.images:
            if image_dir not in seen:
                seen.add(image_dir)
                yield image_dir

    def cuda_graph_enabled(self) -> bool:
        """Return whether CUDA graph replay should be enabled."""

        return self.tensorrt.fp16 and not self.debug.verbose


__all__ = [
    "DatasetPaths",
    "TensorRTBuildConfig",
    "QuantizationConfig",
    "PreprocessingConfig",
    "PostprocessingConfig",
    "TrackerConfig",
    "ProfilingConfig",
    "DeploymentPaths",
    "DebugConfig",
    "YOLOv8Config",
]
