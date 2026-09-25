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

# CUDA major versions this project supports, in order of preference
# (newest first).  CuPy ships a separate wheel per major version
# (``cupy-cuda12x`` for CUDA 12.x, ``cupy-cuda13x`` for CUDA 13.x), so a
# usable install is one where the CuPy wheel matches the available CUDA
# Toolkit major version.
_SUPPORTED_CUDA_MAJORS: tuple[int, ...] = (13, 12)


def _cupy_install_hint() -> str:
    """Return a pip install hint listing the supported CuPy wheels."""
    wheels = " or ".join(f"cupy-cuda{major}x" for major in _SUPPORTED_CUDA_MAJORS)
    return f"install the wheel matching your CUDA Toolkit, e.g. pip install {wheels}"


def _major_pref_rank(major: int) -> tuple[int, int]:
    """Sort key ranking CUDA majors: supported first, then newest first."""
    if major in _SUPPORTED_CUDA_MAJORS:
        return (0, -major)
    # Unknown/unsupported majors come last, newest first.
    return (1, -major)


def _cuda_path_var_major(key: str) -> int:
    """Parse the major version from a ``CUDA_PATH_V<major>_<minor>`` key.

    For example ``CUDA_PATH_V12_6`` -> 12 and ``CUDA_PATH_V13_0`` -> 13.
    Returns 0 when no version can be parsed.
    """
    suffix = key[len("CUDA_PATH_V"):]
    digits = ""
    for ch in suffix:
        if ch.isdigit():
            digits += ch
        else:
            break
    return int(digits) if digits else 0


def _cuda_runtime_major(cupy_mod: Any) -> int:
    """Return the CUDA runtime major version CuPy reports, or 0 on failure.

    ``runtimeGetVersion`` returns an int like ``12060`` (12.6) or
    ``13000`` (13.0); the major version is that value // 1000.
    """
    try:
        version = int(cupy_mod.cuda.runtime.runtimeGetVersion())
    except Exception:
        return 0
    return version // 1000


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
    if _use_cuda_root(os.environ.get("CUDA_PATH"), "CUDA_PATH"):
        return

    # 2. CUDA_HOME (alternative environment variable).
    if _use_cuda_root(os.environ.get("CUDA_HOME"), "CUDA_HOME"):
        return

    # 3. CUDA_PATH_V* variables (set by some CUDA installers, e.g.
    #    CUDA_PATH_V12_6 or CUDA_PATH_V13_0).  Prefer the newest supported
    #    major version (13.x, then 12.x).
    cuda_path_vars = [
        (key, val) for key, val in os.environ.items()
        if key.startswith("CUDA_PATH_V")
    ]
    # Sort by version preference (supported majors first, newest first),
    # then by key name for determinism.
    cuda_path_vars.sort(
        key=lambda kv: (_major_pref_rank(_cuda_path_var_major(kv[0])), kv[0])
    )
    for key, val in cuda_path_vars:
        if _use_cuda_root(val, key):
            return

    # 4. Conda environment — cudatoolkit packages install into
    #    <env>/Library/ on Windows.
    conda_library = Path(sys.prefix) / "Library"
    if _has_nvrtc(conda_library) and _use_cuda_root(
        str(conda_library), "conda env"
    ):
        return

    # 5. Check if nvcc is already on PATH and derive root from it.
    #    nvcc lives in <cuda_root>/bin/nvcc.exe (the root is normalised,
    #    so a copy under bin/x64 also resolves correctly).
    nvcc_path = _find_nvcc_on_path()
    if nvcc_path is not None and _use_cuda_root(str(nvcc_path.parent), "nvcc"):
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
        # Highest-versioned directories first (e.g. v13.0 before v12.6).
        versions = sorted(
            (d for d in root.iterdir() if d.is_dir() and _has_nvrtc(d)),
            key=lambda d: d.name,
            reverse=True,
        )
        # Prefer the newest supported major version (13.x, then 12.x).
        for major in _SUPPORTED_CUDA_MAJORS:
            for ver_dir in versions:
                if ver_dir.name.startswith(f"v{major}"):
                    if _use_cuda_root(str(ver_dir), "Program Files"):
                        return
        # If no supported major was found, try any version as a fallback —
        # CuPy may still work if the installed wheel matches this major.
        for ver_dir in versions:
            if ver_dir.name.startswith("v"):
                logger.info(
                    "CUDA at %s has an unrecognised major version — "
                    "ensure your CuPy wheel matches",
                    ver_dir,
                )
                if _use_cuda_root(str(ver_dir), "Program Files"):
                    return

    logger.warning(
        "Could not auto-detect a CUDA installation on Windows. "
        "Set the CUDA_PATH environment variable to your CUDA Toolkit "
        "directory (e.g. C:\\Program Files\\NVIDIA GPU Computing Toolkit"
        "\\CUDA\\v13.0 or ...\\CUDA\\v12.6)."
    )


def _normalize_cuda_root(path: Path) -> Path:
    """Return the CUDA Toolkit root for *path*.

    Users and installers sometimes point ``CUDA_PATH`` at a subdirectory
    such as ``<root>\\bin`` or ``<root>\\bin\\x64`` instead of the toolkit
    root.  CuPy appends ``bin`` itself, so such a value produces a
    non-existent ``...\\bin\\bin`` directory.  Strip those trailing
    components so the root is always used.
    """
    parts = [p.lower() for p in path.parts]
    if len(parts) >= 2 and parts[-1] == "x64" and parts[-2] in ("bin", "lib"):
        return path.parent.parent
    if parts and parts[-1] == "bin":
        return path.parent
    return path


def _cuda_dll_dirs(cuda_root: Path) -> list[Path]:
    """Return the directories that may hold CUDA DLLs under *cuda_root*.

    CUDA 12.x and older keep DLLs in ``bin``; CUDA 13.x moved them to
    ``bin\\x64``.  Both are returned so either layout works.
    """
    return [
        cuda_root / "bin",
        cuda_root / "bin" / "x64",
        cuda_root / "lib" / "x64",
    ]


def _has_nvrtc(cuda_root: Path) -> bool:
    """Return True if *cuda_root* contains an ``nvrtc64_*.dll``."""
    for dll_dir in _cuda_dll_dirs(cuda_root)[:2]:
        if dll_dir.is_dir() and any(dll_dir.glob("nvrtc64_*.dll")):
            return True
    return False


def _use_cuda_root(value: str | None, source: str) -> bool:
    """Adopt *value* as the CUDA Toolkit root if it is a valid directory.

    The path is normalised to the toolkit root, written back to
    ``CUDA_PATH`` (which CuPy reads) and its DLL directories are added to
    the search path.  Returns True on success.
    """
    if not value:
        return False
    cuda_root = _normalize_cuda_root(Path(value))
    if not (cuda_root / "bin").is_dir():
        logger.debug("Ignoring %s = %s: no bin directory under %s",
                     source, value, cuda_root)
        return False
    if cuda_root != Path(value):
        logger.info("%s = %s is not the toolkit root; using %s",
                    source, value, cuda_root)
    else:
        logger.info("Found CUDA via %s = %s", source, cuda_root)
    os.environ["CUDA_PATH"] = str(cuda_root)
    _add_cuda_bin_to_path(cuda_root)
    return True


def _find_nvcc_on_path() -> Path | None:
    """Return the path to ``nvcc.exe`` if found on PATH, else *None*."""
    import shutil

    nvcc = shutil.which("nvcc")
    if nvcc is not None:
        return Path(nvcc).resolve()
    return None


def _add_cuda_bin_to_path(cuda_root: Path) -> None:
    """Ensure CUDA DLL directories are on the DLL search path.

    Adds ``<cuda_root>/bin`` (nvrtc, cudart, etc. on CUDA 12.x),
    ``<cuda_root>/bin/x64`` (the same DLLs on CUDA 13.x) and
    ``<cuda_root>/lib/x64`` (additional libraries on some installations)
    to PATH and the DLL search directories.
    """
    dirs_to_add = [d for d in _cuda_dll_dirs(cuda_root) if d.is_dir()]
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


def _test_nvrtc(cupy) -> tuple[bool, str]:
    """Run a small computation to verify NVRTC works.

    Returns ``(success, error_message)``.
    """
    try:
        a = cupy.array([1.0, 2.0, 3.0])
        _ = float((a * a).sum())
        return True, ""
    except Exception as exc:
        return False, str(exc)


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
        return False, "", f"CuPy is not installed ({_cupy_install_hint()})"
    except Exception as exc:
        return False, "", f"CuPy import failed: {exc}"

    # Report the CUDA runtime version CuPy is built against, and flag a
    # wheel that targets an unsupported major version up front.
    runtime_major = _cuda_runtime_major(cupy)
    if runtime_major:
        logger.info("CuPy is using CUDA runtime major version %d", runtime_major)
        if runtime_major not in _SUPPORTED_CUDA_MAJORS:
            supported = ", ".join(
                f"{m}.x" for m in sorted(_SUPPORTED_CUDA_MAJORS, reverse=True)
            )
            return (
                False,
                _get_device_name(cupy),
                f"CuPy is built for CUDA {runtime_major}.x, which is not "
                f"supported (supported: {supported}). {_cupy_install_hint()}.",
            )

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
    # kernels if nvrtc DLLs are missing or incompatible.
    ok, err_msg = _test_nvrtc(cupy)

    if not ok and "--std" in err_msg:
        # Some CuPy / CUDA version combinations produce a malformed --std
        # flag (e.g. ``--std`` without a value).  Retry with an explicit
        # C++ standard override.
        for std in ("--std=c++14", "--std=c++11", "--std=c++17"):
            logger.debug("NVRTC --std error, retrying with %s", std)
            os.environ["CUPY_NVRTC_COMPILE_OPTIONS"] = std
            ok, err_msg = _test_nvrtc(cupy)
            if ok:
                logger.info(
                    "NVRTC workaround succeeded with %s", std
                )
                break

    if not ok:
        if "--std" in err_msg:
            return (
                False,
                name,
                f"NVRTC compilation failed: {err_msg}. "
                "This usually indicates a CuPy / CUDA Toolkit version "
                f"mismatch. Reinstall CuPy to match your CUDA Toolkit "
                f"({_cupy_install_hint()}), or install a supported "
                "CUDA Toolkit (12.x or 13.x).",
            )
        if "nvrtc" in err_msg.lower() or "FileNotFoundError" in err_msg:
            return (
                False,
                name,
                f"CUDA toolkit libraries (NVRTC) not found: {err_msg}. "
                "Install a supported CUDA Toolkit (12.x or 13.x) or set the "
                "CUDA_PATH environment variable to your CUDA installation "
                "directory.",
            )
        return False, name, f"GPU computation test failed: {err_msg}"

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
