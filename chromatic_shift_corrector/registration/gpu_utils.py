"""GPU detection and cupy/numpy fallback logic."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_GPU_AVAILABLE: bool | None = None
_GPU_NAME: str = ""
_GPU_FAIL_REASON: str = ""


def _ensure_cuda_env() -> None:
    """Try to locate and configure CUDA paths on Windows.

    If ``CUDA_PATH`` is not set, searches common NVIDIA CUDA Toolkit
    installation directories and sets the environment variable so that
    CuPy can locate libraries like ``nvrtc64_*.dll``.
    """
    if sys.platform != "win32":
        return

    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path and Path(cuda_path).is_dir():
        # CUDA_PATH already set — just ensure bin/ is on PATH.
        _add_cuda_bin_to_path(Path(cuda_path))
        return

    # Search common CUDA installation directories.
    search_roots = [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        / "NVIDIA GPU Computing Toolkit"
        / "CUDA",
    ]

    # Also check CUDA_PATH_V* variables (set by some CUDA installers).
    for key, val in os.environ.items():
        if key.startswith("CUDA_PATH_V") and Path(val).is_dir():
            logger.info("Found CUDA via %s = %s", key, val)
            os.environ["CUDA_PATH"] = val
            _add_cuda_bin_to_path(Path(val))
            return

    for root in search_roots:
        if not root.is_dir():
            continue
        # Pick the highest-versioned CUDA 12.x directory available.
        versions = sorted(root.iterdir(), reverse=True)
        for ver_dir in versions:
            if ver_dir.is_dir() and ver_dir.name.startswith("v12"):
                nvrtc_candidates = list(ver_dir.glob("bin/nvrtc64_*.dll"))
                if nvrtc_candidates:
                    logger.info("Auto-detected CUDA at %s", ver_dir)
                    os.environ["CUDA_PATH"] = str(ver_dir)
                    _add_cuda_bin_to_path(ver_dir)
                    return

    logger.debug("Could not auto-detect a CUDA 12.x installation on Windows.")


def _add_cuda_bin_to_path(cuda_root: Path) -> None:
    """Ensure ``<cuda_root>/bin`` is on the DLL search path."""
    bin_dir = str(cuda_root / "bin")
    current_path = os.environ.get("PATH", "")
    if bin_dir.lower() not in current_path.lower():
        os.environ["PATH"] = bin_dir + os.pathsep + current_path
        logger.debug("Added %s to PATH", bin_dir)

    # On Python 3.8+ / Windows, os.add_dll_directory is needed for DLL
    # resolution in addition to PATH.
    if hasattr(os, "add_dll_directory") and Path(bin_dir).is_dir():
        try:
            os.add_dll_directory(bin_dir)
        except OSError:
            pass


def _probe_gpu() -> tuple[bool, str, str]:
    """Try to import cupy and detect a suitable NVIDIA GPU.

    Returns ``(available, gpu_name, fail_reason)``.
    """
    # Attempt to fix CUDA path before importing CuPy.
    try:
        _ensure_cuda_env()
    except Exception as exc:
        logger.debug("CUDA env setup failed: %s", exc)

    try:
        import cupy  # noqa: F401
    except ImportError:
        return False, "", "CuPy is not installed (install with: pip install cupy-cuda12x)"
    except Exception as exc:
        return False, "", f"CuPy import failed: {exc}"

    try:
        dev = cupy.cuda.Device(0)
        cc = dev.compute_capability
        # Require compute capability >= 8.6 (RTX 3060+).
        cc_int = int(cc)
        if cc_int < 86:
            name = _get_device_name(cupy)
            return (
                False,
                name,
                f"GPU compute capability {cc} is below the minimum 8.6 required",
            )
        name = _get_device_name(cupy)
    except Exception as exc:
        return False, "", f"CUDA device detection failed: {exc}"

    # Verify that NVRTC (runtime compiler) actually works.  CuPy can
    # detect the GPU via the CUDA driver but fail later when JIT-compiling
    # kernels if nvrtc DLLs are missing.
    try:
        a = cupy.array([1.0, 2.0, 3.0])
        _ = float((a * a).sum())
    except Exception as exc:
        msg = str(exc)
        if "nvrtc" in msg.lower() or "FileNotFoundError" in msg:
            return (
                False,
                name,
                f"CUDA toolkit libraries (NVRTC) not found: {msg}. "
                "Install the CUDA 12.x Toolkit or set the CUDA_PATH "
                "environment variable to your CUDA installation directory.",
            )
        return False, name, f"GPU computation test failed: {exc}"

    return True, name, ""


def _get_device_name(cupy_mod: Any) -> str:
    """Extract the GPU device name, returning empty string on failure."""
    try:
        name = cupy_mod.cuda.runtime.getDeviceProperties(0)["name"]
        if isinstance(name, bytes):
            name = name.decode()
        return name
    except Exception:
        return ""


def gpu_available() -> bool:
    """Return True if a suitable GPU + cupy installation is detected."""
    global _GPU_AVAILABLE, _GPU_NAME, _GPU_FAIL_REASON
    if _GPU_AVAILABLE is None:
        _GPU_AVAILABLE, _GPU_NAME, _GPU_FAIL_REASON = _probe_gpu()
        if _GPU_AVAILABLE:
            logger.info("GPU enabled: %s", _GPU_NAME)
        elif _GPU_FAIL_REASON:
            logger.warning("GPU unavailable: %s", _GPU_FAIL_REASON)
    return _GPU_AVAILABLE


def gpu_name() -> str:
    """Return the GPU device name, or empty string if unavailable."""
    gpu_available()  # ensure probed
    return _GPU_NAME


def gpu_fail_reason() -> str:
    """Return a human-readable reason the GPU is unavailable, or empty string."""
    gpu_available()  # ensure probed
    return _GPU_FAIL_REASON


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
