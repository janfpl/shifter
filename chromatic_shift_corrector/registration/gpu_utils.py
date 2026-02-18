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

    Search order:
    1. ``CUDA_PATH`` environment variable (already set)
    2. ``CUDA_HOME`` environment variable (alternative convention)
    3. ``CUDA_PATH_V*`` environment variables (set by CUDA installers)
    4. Conda environment ``Library`` directory (conda-installed toolkit)
    5. ``nvcc`` on PATH (derive root from its location)
    6. Standard NVIDIA install directories under Program Files
    """
    if sys.platform != "win32":
        return

    # 1. CUDA_PATH already set.
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path and Path(cuda_path).is_dir():
        _add_cuda_bin_to_path(Path(cuda_path))
        return

    # 2. CUDA_HOME (alternative environment variable).
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home and Path(cuda_home).is_dir():
        logger.info("Found CUDA via CUDA_HOME = %s", cuda_home)
        os.environ["CUDA_PATH"] = cuda_home
        _add_cuda_bin_to_path(Path(cuda_home))
        return

    # 3. CUDA_PATH_V* variables (set by some CUDA installers, e.g.
    #    CUDA_PATH_V12_6).  Prefer 12.x versions.
    cuda_path_vars: list[tuple[str, str]] = []
    for key, val in os.environ.items():
        if key.startswith("CUDA_PATH_V") and Path(val).is_dir():
            cuda_path_vars.append((key, val))
    # Sort so that CUDA_PATH_V12* entries come first (preferred).
    cuda_path_vars.sort(key=lambda kv: (not kv[0].startswith("CUDA_PATH_V12"), kv[0]))
    for key, val in cuda_path_vars:
        logger.info("Found CUDA via %s = %s", key, val)
        os.environ["CUDA_PATH"] = val
        _add_cuda_bin_to_path(Path(val))
        return

    # 4. Conda environment — cudatoolkit packages install into
    #    <env>/Library/ on Windows.
    conda_library = Path(sys.prefix) / "Library"
    if conda_library.is_dir():
        # Check for nvrtc DLLs in the conda env's bin directory.
        conda_bin = conda_library / "bin"
        if conda_bin.is_dir() and list(conda_bin.glob("nvrtc64_*.dll")):
            logger.info("Auto-detected CUDA in conda env at %s", conda_library)
            os.environ["CUDA_PATH"] = str(conda_library)
            _add_cuda_bin_to_path(conda_library)
            return

    # 5. Check if nvcc is already on PATH and derive root from it.
    nvcc_path = _find_nvcc_on_path()
    if nvcc_path is not None:
        # nvcc lives in <cuda_root>/bin/nvcc.exe
        cuda_root = nvcc_path.parent.parent
        logger.info("Auto-detected CUDA via nvcc at %s", cuda_root)
        os.environ["CUDA_PATH"] = str(cuda_root)
        _add_cuda_bin_to_path(cuda_root)
        return

    # 6. Search common CUDA installation directories on the filesystem.
    search_roots = []
    for env_var in ("ProgramFiles", "ProgramFiles(x86)"):
        pf = os.environ.get(env_var)
        if pf:
            search_roots.append(
                Path(pf) / "NVIDIA GPU Computing Toolkit" / "CUDA"
            )
    # Fallback if env vars are missing.
    if not search_roots:
        search_roots.append(
            Path(r"C:\Program Files") / "NVIDIA GPU Computing Toolkit" / "CUDA"
        )

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
        # If no v12.x found, try any version as a fallback — CuPy may
        # still work if the major version is compatible.
        for ver_dir in versions:
            if ver_dir.is_dir() and ver_dir.name.startswith("v"):
                nvrtc_candidates = list(ver_dir.glob("bin/nvrtc64_*.dll"))
                if nvrtc_candidates:
                    logger.info(
                        "Auto-detected CUDA at %s (non-12.x — may not "
                        "be compatible with cupy-cuda12x)",
                        ver_dir,
                    )
                    os.environ["CUDA_PATH"] = str(ver_dir)
                    _add_cuda_bin_to_path(ver_dir)
                    return

    logger.warning(
        "Could not auto-detect a CUDA installation on Windows. "
        "Set the CUDA_PATH environment variable to your CUDA Toolkit "
        "directory (e.g. C:\\Program Files\\NVIDIA GPU Computing Toolkit"
        "\\CUDA\\v12.6)."
    )


def _find_nvcc_on_path() -> Path | None:
    """Return the path to ``nvcc.exe`` if found on PATH, else *None*."""
    import shutil

    nvcc = shutil.which("nvcc")
    if nvcc is not None:
        return Path(nvcc).resolve()
    return None


def _add_cuda_bin_to_path(cuda_root: Path) -> None:
    """Ensure CUDA DLL directories are on the DLL search path.

    Adds both ``<cuda_root>/bin`` (contains nvrtc, cudart, etc.) and
    ``<cuda_root>/lib/x64`` (contains additional libraries on some
    installations) to PATH and the DLL search directories.
    """
    dirs_to_add = [
        cuda_root / "bin",
        cuda_root / "lib" / "x64",
    ]
    current_path = os.environ.get("PATH", "")
    for dll_dir in dirs_to_add:
        dll_dir_str = str(dll_dir)
        if dll_dir_str.lower() not in current_path.lower():
            os.environ["PATH"] = dll_dir_str + os.pathsep + os.environ.get("PATH", "")
            logger.debug("Added %s to PATH", dll_dir_str)

        # On Python 3.8+ / Windows, os.add_dll_directory is needed for
        # DLL resolution in addition to PATH.
        if hasattr(os, "add_dll_directory") and dll_dir.is_dir():
            try:
                os.add_dll_directory(str(dll_dir))
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
