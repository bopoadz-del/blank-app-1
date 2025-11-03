"""High-level TensorRT inference session for YOLOv8."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

from .config import PostprocessingConfig, PreprocessingConfig, YOLOv8Config
from .memory import BufferBinding, BufferPool, StagingArea, create_bindings
from .postprocessing import BatchDetections, postprocess
from .preprocessing import BatchBuilder, Letterbox, Normalizer

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import numpy as np
except Exception:  # pragma: no cover - fallback
    np = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import tensorrt as trt  # type: ignore
except Exception:  # pragma: no cover - fallback
    trt = None  # type: ignore


@dataclass
class EngineIO:
    bindings: List[BufferBinding]
    host_inputs: List["np.ndarray"]
    host_outputs: List["np.ndarray"]


class TensorRTInferenceSession:
    """Encapsulates TensorRT execution context and bindings."""

    def __init__(
        self,
        config: YOLOv8Config,
        engine_loader: Callable[[], bytes],
        class_names: Sequence[str],
    ) -> None:
        self.config = config
        self.engine_loader = engine_loader
        self.class_names = list(class_names)
        self.preprocess = self._build_preprocessor(config.preprocessing)
        self.postprocess_config = config.postprocessing
        self.engine_bytes: Optional[bytes] = None
        self.runtime = None
        self.engine = None
        self.context = None
        self.bindings: Optional[EngineIO] = None

    def _build_preprocessor(self, cfg: PreprocessingConfig) -> BatchBuilder:
        letterbox = Letterbox(cfg.input_width, cfg.input_height, cfg.padding_color)
        normalizer = Normalizer(cfg.mean, cfg.std) if cfg.normalize else None
        return BatchBuilder(letterbox, normalizer, cfg.use_pinned_memory)

    def load_engine(self) -> None:
        if trt is None:
            LOGGER.warning("TensorRT runtime unavailable; inference disabled")
            return
        self.engine_bytes = self.engine_loader()
        runtime = trt.Runtime(trt.Logger())
        engine = runtime.deserialize_cuda_engine(self.engine_bytes)
        context = engine.create_execution_context()
        self.runtime = runtime
        self.engine = engine
        self.context = context
        self.bindings = self._create_bindings(engine)

    def _create_bindings(self, engine) -> EngineIO:
        if np is None:
            raise RuntimeError("NumPy required for inference")
        bindings: List[BufferBinding] = []
        host_inputs: List["np.ndarray"] = []
        host_outputs: List["np.ndarray"] = []
        binding_shapes = []
        for index in range(engine.num_bindings):
            shape = engine.get_binding_shape(index)
            dtype = trt.nptype(engine.get_binding_dtype(index))
            binding_shapes.append((shape, dtype))
        for shape, dtype in binding_shapes:
            for binding in create_bindings([tuple(int(dim) for dim in shape)], dtype):
                bindings.append(binding)
                array = binding.host.array.view(dtype).reshape(shape)
                if len(host_inputs) < engine.num_io_tensors:  # heuristics for demo
                    host_inputs.append(array)
                else:
                    host_outputs.append(array)
        return EngineIO(bindings=bindings, host_inputs=host_inputs, host_outputs=host_outputs)

    def infer(self, images: Sequence["np.ndarray"]) -> List[BatchDetections]:
        if np is None:
            raise RuntimeError("NumPy required for inference")
        if self.context is None or self.bindings is None:
            raise RuntimeError("Engine not loaded")
        batch, stats = self.preprocess.build_batch(images)
        outputs = np.zeros_like(self.bindings.host_outputs[0])
        # In a real implementation we would enqueue the buffers on CUDA streams
        outputs[:] = batch.mean(axis=0)  # placeholder to keep shapes consistent
        return postprocess(
            outputs[None, ...],
            stats,
            anchors=np.ones((3, 2), dtype=np.float32),
            score_threshold=self.postprocess_config.score_threshold,
            iou_threshold=self.postprocess_config.nms_iou_threshold,
            max_detections=self.postprocess_config.max_detections,
            class_names=self.class_names,
            multi_label=self.postprocess_config.multi_label,
        )


__all__ = ["TensorRTInferenceSession"]
