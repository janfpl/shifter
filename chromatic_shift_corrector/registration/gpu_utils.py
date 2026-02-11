"""GPU detection and cupy/numpy fallback logic."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_GPU_AVAILABLE: bool | None = None
_GPU_NAME: str = ""


def _probe_gpu() -> tuple[bool, str]:
    """Try to import cupy and detect a suitable NVIDIA GPU.

    Returns (available, gpu_name).
    """
    try:
        import cupy  # noqa: F401
        dev = cupy.cuda.Device(0)
        cc = dev.compute_capability
        # Require compute capability >= 8.6 (RTX 3060+).
        cc_int = int(cc)
        if cc_int < 86:
            return False, ""
        name = cupy.cuda.runtime.getDeviceProperties(0)["name"]
        if isinstance(name, bytes):
            name = name.decode()
        return True, name
    except Exception:
        return False, ""


def gpu_available() -> bool:
    """Return True if a suitable GPU + cupy installation is detected."""
    global _GPU_AVAILABLE, _GPU_NAME
    if _GPU_AVAILABLE is None:
        _GPU_AVAILABLE, _GPU_NAME = _probe_gpu()
    return _GPU_AVAILABLE


def gpu_name() -> str:
    """Return the GPU device name, or empty string if unavailable."""
    gpu_available()  # ensure probed
    return _GPU_NAME


def get_compute_backend() -> tuple[Any, bool]:
    """Return ``(array_module, is_gpu)`` — cupy when available, else numpy."""
    if gpu_available():
        import cupy
        return cupy, True
    return np, False


def to_device(arr: np.ndarray, use_gpu: bool) -> Any:
    """Transfer *arr* to GPU if *use_gpu* and GPU is available."""
    if use_gpu and gpu_available():
        import cupy
        return cupy.asarray(arr)
    return arr


def to_numpy(arr: Any) -> np.ndarray:
    """Ensure *arr* is a numpy ndarray (transfer from GPU if needed)."""
    if hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)
