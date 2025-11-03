"""CUDA-aware memory helpers used by the YOLOv8 TensorRT runtime.

The focus of this module is to provide zero-copy transfer primitives that keep
GPU and CPU buffers in sync without redundant allocations.  Inference
performance at 270 FPS demands careful management of CUDA streams and page-locked
buffers, so the utilities here abstract the lower level PyCUDA (or CuPy/NVIDIA
CUDA Python) APIs behind simple classes.  The implementations gracefully degrade
when CUDA is unavailable so unit tests can exercise the logic on CPU-only
machines.
"""

from __future__ import annotations

import logging
import queue
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator, Iterable, Optional, Tuple

try:  # pragma: no cover - optional dependency
    import numpy as np
except Exception:  # pragma: no cover - testing fallback
    np = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import cuda  # type: ignore
except Exception:  # pragma: no cover - testing fallback
    cuda = None  # type: ignore

LOGGER = logging.getLogger(__name__)


class CUDAError(RuntimeError):
    """Raised when CUDA operations fail and cannot be recovered from."""


@dataclass
class DeviceBuffer:
    """Represents a GPU buffer backed by CUDA device memory."""

    ptr: int
    size: int
    stream: Optional["cuda.Stream"] = None

    def free(self) -> None:
        if cuda is None:  # pragma: no cover - CPU fallback
            return
        cuda.cuMemFree(self.ptr)


@dataclass
class HostBuffer:
    """Represents a page-locked host buffer."""

    array: "np.ndarray"
    size: int

    def free(self) -> None:
        if cuda is None:  # pragma: no cover - CPU fallback
            return
        cuda.cuMemHostUnregister(self.array.ctypes.data)


def _cuda_driver() -> Optional[object]:  # pragma: no cover - runtime helper
    return cuda


def allocate_device(size: int) -> DeviceBuffer:
    """Allocate device memory returning a :class:`DeviceBuffer`."""

    driver = _cuda_driver()
    if driver is None:
        raise CUDAError("CUDA driver is unavailable; cannot allocate device memory")
    ptr = driver.cuMemAlloc(size)
    return DeviceBuffer(ptr=ptr, size=size)


def allocate_host(size: int) -> HostBuffer:
    """Allocate pinned host memory returning a :class:`HostBuffer`."""

    if np is None:
        raise CUDAError("NumPy is required for host allocations")
    buffer = np.empty(size, dtype=np.uint8)
    driver = _cuda_driver()
    if driver is not None:  # pragma: no branch - negligible impact
        driver.cuMemHostRegister(buffer.ctypes.data, size, 0)
    return HostBuffer(array=buffer, size=size)


def memcpy_htod(dest: DeviceBuffer, src: HostBuffer, size: Optional[int] = None) -> None:
    """Copy memory from host to device using the associated CUDA stream."""

    driver = _cuda_driver()
    if driver is None:
        raise CUDAError("CUDA driver unavailable for host to device copy")
    transfer_size = size or min(dest.size, src.size)
    driver.cuMemcpyHtoDAsync(dest.ptr, src.array.ctypes.data, transfer_size, dest.stream or 0)


def memcpy_dtoh(dest: HostBuffer, src: DeviceBuffer, size: Optional[int] = None) -> None:
    """Copy memory from device to host using the associated CUDA stream."""

    driver = _cuda_driver()
    if driver is None:
        raise CUDAError("CUDA driver unavailable for device to host copy")
    transfer_size = size or min(dest.size, src.size)
    driver.cuMemcpyDtoHAsync(dest.array.ctypes.data, src.ptr, transfer_size, src.stream or 0)


def synchronize(stream: Optional["cuda.Stream"] = None) -> None:
    """Synchronise the provided stream or the default stream when ``None``."""

    driver = _cuda_driver()
    if driver is None:
        return
    (stream or driver.Stream.null).synchronize()


@dataclass
class BufferBinding:
    """Binds a host and device buffer pair for zero-copy workflows."""

    host: HostBuffer
    device: DeviceBuffer
    shape: Tuple[int, ...]
    dtype: "np.dtype"

    def upload(self) -> None:
        memcpy_htod(self.device, self.host)

    def download(self) -> None:
        memcpy_dtoh(self.host, self.device)


class BufferPool:
    """Thread-safe pool of reusable :class:`BufferBinding` instances."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._pool: "queue.Queue[BufferBinding]" = queue.Queue(maxsize=capacity)
        self._lock = threading.Lock()
        LOGGER.debug("BufferPool initialised with capacity %d", capacity)

    def put(self, binding: BufferBinding) -> None:
        LOGGER.debug("Returning buffer %s to pool", binding)
        self._pool.put(binding)

    def get(self, timeout: Optional[float] = None) -> BufferBinding:
        LOGGER.debug("Requesting buffer from pool with timeout %s", timeout)
        return self._pool.get(timeout=timeout)

    def qsize(self) -> int:
        return self._pool.qsize()

    def empty(self) -> bool:
        return self._pool.empty()

    def clear(self) -> None:
        LOGGER.debug("Clearing buffer pool")
        while not self._pool.empty():
            binding = self._pool.get_nowait()
            binding.device.free()
            binding.host.free()


def create_bindings(shapes: Iterable[Tuple[int, ...]], dtype: "np.dtype") -> Iterable[BufferBinding]:
    """Create buffer bindings for the provided shapes."""

    if np is None:
        raise CUDAError("NumPy is required to create buffer bindings")

    for shape in shapes:
        size = int(np.prod(shape)) * dtype.itemsize
        host = allocate_host(size)
        device = allocate_device(size)
        array = np.frombuffer(host.array, dtype=dtype)
        array = array.reshape(shape)
        yield BufferBinding(host=host, device=device, shape=shape, dtype=dtype)


@contextmanager
def stream_guard(stream: Optional["cuda.Stream"]) -> Generator["cuda.Stream", None, None]:
    """Context manager that temporarily sets the active CUDA stream."""

    driver = _cuda_driver()
    if driver is None or stream is None:
        yield stream  # type: ignore[misc]
        return

    prev = driver.get_current_stream()
    driver.set_current_stream(stream)
    try:
        yield stream
    finally:
        driver.set_current_stream(prev)


class StagingArea:
    """High-level helper for staging batches into GPU memory with minimal copies."""

    def __init__(self, pool: BufferPool, batch_shape: Tuple[int, ...], dtype: "np.dtype") -> None:
        self.pool = pool
        self.batch_shape = batch_shape
        self.dtype = dtype
        self._active: Optional[BufferBinding] = None

    def __enter__(self) -> "np.ndarray":
        binding = self.pool.get()
        self._active = binding
        return binding.host.array.view(self.dtype).reshape(self.batch_shape)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._active is None:
            return
        try:
            if exc is None:
                self._active.upload()
        finally:
            self.pool.put(self._active)
            self._active = None


__all__ = [
    "CUDAError",
    "DeviceBuffer",
    "HostBuffer",
    "BufferBinding",
    "BufferPool",
    "StagingArea",
    "allocate_device",
    "allocate_host",
    "memcpy_htod",
    "memcpy_dtoh",
    "synchronize",
    "stream_guard",
    "create_bindings",
]
