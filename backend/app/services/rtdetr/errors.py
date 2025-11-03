"""Error hierarchy for the RT-DETR deployment stack.

Centralized exceptions simplify debugging workflows during deployment,
particularly when integrating with orchestration services that need to
classify failures into actionable categories.
"""

from __future__ import annotations

import textwrap
from typing import Optional


class RTDETRError(RuntimeError):
    """Base error for all RT-DETR related failures."""

    def __init__(self, message: str, *, hint: Optional[str] = None) -> None:
        final_message = message
        if hint:
            final_message = f"{message}\n{self._format_hint(hint)}"
        super().__init__(final_message)
        self.message = message
        self.hint = hint

    @staticmethod
    def _format_hint(hint: str) -> str:
        return textwrap.indent(f"Hint: {hint}", prefix="  ")


class RTDETRConfigurationError(RTDETRError):
    """Raised when configuration settings are incompatible."""


class RTDETRExecutionError(RTDETRError):
    """Raised when runtime execution fails."""


class RTDETRTimeoutError(RTDETRExecutionError):
    """Raised when the inference loop exceeds the configured deadline."""


class RTDETRBuilderError(RTDETRError):
    """Raised when TensorRT engine construction fails."""


class RTDETRCalibrationError(RTDETRError):
    """Raised when INT8 calibration is unable to converge."""


class RTDETRStreamError(RTDETRError):
    """Raised when multi-stream orchestration encounters a fatal error."""


class RTDETRProfilingError(RTDETRError):
    """Raised when profiling hooks fail to collect measurements."""


class RTDETRBatchingError(RTDETRError):
    """Raised when batch orchestration fails to schedule work."""


__all__ = [
    "RTDETRError",
    "RTDETRConfigurationError",
    "RTDETRExecutionError",
    "RTDETRTimeoutError",
    "RTDETRBuilderError",
    "RTDETRCalibrationError",
    "RTDETRStreamError",
    "RTDETRProfilingError",
    "RTDETRBatchingError",
]
