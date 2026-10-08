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

# CUDA runtime/driver versions detected during probing, if any:
# ``{"runtime": int, "driver": int}`` (CuPy's integer form, e.g. 12060 = 12.6).
_GPU_CUDA_VERSIONS: dict[str, int] | None = None

# CUDA major versions this project supports, in order of preference (newest
# first). CuPy ships a separate wheel per major version (``cupy-cuda12x`` for
# CUDA 12.x, ``cupy-cuda13x`` for CUDA 13.x); each works with any minor version
# of its major line (the runtime may report e.g. 12.9 even when the separately
# installed toolkit is 12.6 — that is fine). 12.6 is just the version this app
# is tested against.
_SUPPORTED_CUDA_MAJORS: tuple[int, ...] = (13, 12)
_SUPPORTED_CUDA_LINE = " or ".join(f"{m}.x" for m in sorted(_SUPPORTED_CUDA_MAJORS))
_TESTED_CUDA = "12.6"

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


# stdout marker the child probe prints (before the crash-prone NVRTC test) so
# the parent can report the CUDA version even when the child faults natively.
_CUDA_MARKER = "##SHIFTER_CUDA_VERSIONS##"


# Probe strategies, tried in order by :func:`_detect_gpu`. Each controls how the
# DLL search path is arranged before CuPy is imported and NVRTC is exercised:
#   "isolated" — remove system CUDA-toolkit dirs from PATH so CuPy uses only its
#                own bundled libraries (fixes a system nvrtc shadowing the
#                bundled one, the common Windows crash).
#   "system"   — inject the system CUDA toolkit path (for conda cudatoolkit and
#                other setups whose CuPy relies on system libraries).
#   "bundled"  — leave PATH untouched (CuPy's default resolution).
_STRATEGIES = ("isolated", "system")

import re as _re

# Matches a system CUDA Toolkit install dir, e.g.
# ``C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin``.
_SYSTEM_CUDA_RE = _re.compile(r"NVIDIA GPU Computing Toolkit[\\/]+CUDA", _re.IGNORECASE)


def _strip_system_cuda_from_path() -> list[str]:
    """Remove system CUDA-toolkit directories from this process's ``PATH``.

    Returns the removed entries (for logging). This lets CuPy load its own
    bundled CUDA libraries instead of a system toolkit's DLLs, which — when the
    two versions differ — otherwise shadow the bundled ``nvrtc`` and crash the
    kernel compile natively.
    """
    if sys.platform != "win32":
        return []
    entries = os.environ.get("PATH", "").split(os.pathsep)
    kept: list[str] = []
    removed: list[str] = []
    for entry in entries:
        (removed if entry and _SYSTEM_CUDA_RE.search(entry) else kept).append(entry)
    if removed:
        os.environ["PATH"] = os.pathsep.join(kept)
    return removed


def _apply_cuda_strategy(strategy: str) -> None:
    """Arrange the DLL search path for a probe *strategy* (see ``_STRATEGIES``)."""
    if strategy == "isolated":
        removed = _strip_system_cuda_from_path()
        if removed:
            logger.debug("Isolated strategy removed %d system CUDA PATH entrie(s)", len(removed))
    elif strategy == "system":
        try:
            _ensure_cuda_env()
        except Exception as exc:
            logger.debug("CUDA env setup failed: %s", exc)
    # "bundled": leave PATH as-is.


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


def _probe_gpu(strategy: str = "isolated") -> tuple[bool, str, str]:
    """Try to import cupy and detect a suitable NVIDIA GPU.

    Parameters
    ----------
    strategy : str
        How to arrange the DLL search path before probing — one of
        ``_STRATEGIES`` (see :func:`_apply_cuda_strategy`). ``"isolated"``
        (the default) removes system CUDA-toolkit dirs from ``PATH`` so CuPy
        uses its own bundled libraries, avoiding the common Windows crash where
        a system ``nvrtc`` DLL shadows the bundled one.

    Returns ``(available, gpu_name, fail_reason)``.
    """
    _apply_cuda_strategy(strategy)

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


def _format_cuda_version(v: int | None) -> str:
    """Format CuPy's integer CUDA version (e.g. 12060) as ``"12.6"``."""
    try:
        v = int(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "unknown"
    return f"{v // 1000}.{(v % 1000) // 10}"


def _read_cuda_versions(strategy: str = "isolated") -> dict[str, int] | None:
    """Query the installed CUDA runtime/driver versions (no kernel compile).

    These are plain driver-API lookups — unlike the NVRTC kernel compile they do
    not JIT anything, so they are safe to call even on a mismatched install.
    Returns ``None`` if CuPy/CUDA is not importable. *strategy* mirrors
    :func:`_probe_gpu` so the version is read under the same DLL search path as
    the probe it accompanies.
    """
    _apply_cuda_strategy(strategy)
    try:
        import cupy
        return {
            "runtime": int(cupy.cuda.runtime.runtimeGetVersion()),
            "driver": int(cupy.cuda.runtime.driverGetVersion()),
        }
    except Exception:
        return None


def _emit_cuda_version_marker(strategy: str = "isolated") -> None:
    """Print the CUDA-version marker line (called by the child probe only)."""
    import json

    info = _read_cuda_versions(strategy)
    if info:
        print(f"{_CUDA_MARKER} {json.dumps(info)}", flush=True)


def _parse_cuda_marker(stdout: str) -> dict[str, int] | None:
    """Extract the CUDA-version marker emitted by the child probe, if present."""
    import json

    for line in stdout.splitlines():
        if line.startswith(_CUDA_MARKER):
            try:
                return json.loads(line[len(_CUDA_MARKER):].strip())
            except ValueError:
                return None
    return None


def _cuda_major(versions: dict[str, int] | None) -> int | None:
    """Major CUDA version from a detected version dict, or None if unknown."""
    if versions and versions.get("runtime"):
        return versions["runtime"] // 1000
    return None


def _nvrtc_failed(reason: str) -> bool:
    """Whether *reason* describes an NVRTC/kernel-compile failure or crash."""
    r = (reason or "").lower()
    return any(
        s in r
        for s in (
            "crashed while testing cupy/nvrtc",
            "gpu probe crashed",
            "nvrtc compilation failed",
            "nvrtc",
        )
    )


def _print_banner(lines: list[str]) -> None:
    """Log a bordered multi-line banner (blank entries dropped)."""
    bar = "=" * 64
    body = [f"  {ln}" for ln in lines if ln]
    logger.warning("\n".join(["", bar, *body, bar]))


def _driver_insufficient(versions: dict[str, int] | None, reason: str) -> bool:
    """Whether the NVIDIA driver is too old for the CUDA runtime CuPy uses.

    ``driverGetVersion`` reports the newest CUDA version the installed driver
    supports. CUDA allows minor-version compatibility within a major line, so
    the driver is only too old when its major version is below the runtime's
    (e.g. a CUDA 12.x driver running ``cupy-cuda13x``).
    """
    if "insufficientdriver" in (reason or "").lower().replace(" ", ""):
        return True
    if versions and versions.get("runtime") and versions.get("driver"):
        return versions["driver"] // 1000 < versions["runtime"] // 1000
    return False


def _report_gpu_unavailable(reason: str) -> None:
    """Log why the GPU is off, distinguishing the common causes clearly.

    1. An NVIDIA driver too old for the CUDA runtime CuPy was built for
       (typically ``cupy-cuda13x`` on a driver older than R580).
    2. A genuinely unsupported CUDA *major* version (not 12.x or 13.x).
    3. A CuPy/NVRTC compile failure on an otherwise-supported runtime —
       almost always a CUDA DLL/PATH conflict (a system toolkit's ``nvrtc``
       shadowing CuPy's own), not a version problem.

    Anything else keeps the plain one-line warning.
    """
    versions = _GPU_CUDA_VERSIONS
    major = _cuda_major(versions)

    if _driver_insufficient(versions, reason):
        lines = ["GPU acceleration DISABLED — NVIDIA driver is too old"]
        if versions and versions.get("runtime") and versions.get("driver"):
            lines.append(
                f"CuPy needs CUDA {_format_cuda_version(versions['runtime'])}, but the "
                f"driver supports up to CUDA {_format_cuda_version(versions['driver'])}."
            )
        lines += [
            "Running on CPU. Either:",
            "  - update your NVIDIA driver (CUDA 13.x needs R580 or newer), or",
            "  - switch to the CUDA 12 build of CuPy:",
            "      pip uninstall cupy-cuda13x && pip install cupy-cuda12x",
        ]
        _print_banner(lines)
        return

    if major is not None and major not in _SUPPORTED_CUDA_MAJORS:
        detected = f"Detected CUDA runtime {_format_cuda_version(versions['runtime'])}."
        _print_banner(
            [
                "GPU acceleration DISABLED — unsupported CUDA version",
                f"Only CUDA {_SUPPORTED_CUDA_LINE} is supported "
                f"(tested with {_TESTED_CUDA}).",
                detected,
                "Running on CPU. Install the CuPy build matching your CUDA",
                "Toolkit (pip install cupy-cuda12x or cupy-cuda13x),",
                "or set SHIFTER_DISABLE_GPU=1.",
            ]
        )
        return

    if _nvrtc_failed(reason):
        detected = ""
        if versions and versions.get("runtime"):
            detected = (
                f"Detected CUDA runtime {_format_cuda_version(versions['runtime'])}"
                " — a supported version, so this is not a version problem."
            )
        wheel = f"cupy-cuda{major}x" if major in _SUPPORTED_CUDA_MAJORS else "cupy-cuda12x"
        _print_banner(
            [
                "GPU acceleration DISABLED — CuPy could not compile a test kernel",
                detected,
                "This is almost always a CUDA DLL/PATH conflict: a system CUDA",
                "toolkit whose nvrtc DLL shadows the one CuPy expects.",
                "Running on CPU. Things to try, in order:",
                f"  - pip install -U {wheel}   (refresh CuPy's CUDA libs)",
                "  - remove any system CUDA '...\\bin' from PATH in this shell",
                "  - update your NVIDIA driver",
                "  - set SHIFTER_DISABLE_GPU=1 to silence this check",
            ]
        )
        return

    logger.warning("GPU unavailable: %s", reason)


# How long to wait for the out-of-process GPU probe before giving up (seconds).
_PROBE_TIMEOUT_S = 30


def _probe_gpu_subprocess(strategy: str = "isolated") -> tuple[bool, str, str]:
    """Run :func:`_probe_gpu` in a child process and return its result.

    The probe imports CuPy and JIT-compiles a test kernel through NVRTC. On a
    machine whose CuPy build does not match the installed CUDA driver/toolkit,
    that compile can trigger a **native** fault (e.g. a Windows access
    violation) that a Python ``try/except`` cannot catch — it would take the
    whole application down during startup. Running it in a subprocess contains
    the blast radius: a crashing child just yields a non-zero exit code, which
    we translate into "GPU unavailable" and keep running on CPU.

    *strategy* (see :func:`_probe_gpu`) is forwarded to the child as
    ``--strategy <name>``.
    """
    import json
    import subprocess

    cmd = [
        sys.executable,
        "-X",
        "faulthandler",
        "-m",
        "shifter.registration._gpu_probe",
        "--strategy",
        strategy,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return (
            False,
            "",
            "GPU probe timed out (possible driver/toolkit hang); using CPU. "
            "Set SHIFTER_DISABLE_GPU=1 to skip the probe.",
        )
    except Exception as exc:
        return False, "", f"GPU probe could not be launched ({exc}); using CPU"

    global _GPU_CUDA_VERSIONS
    _GPU_CUDA_VERSIONS = _parse_cuda_marker(proc.stdout or "")

    if proc.returncode != 0:
        stderr_tail = ""
        if proc.stderr:
            logger.debug("GPU probe subprocess stderr:\n%s", proc.stderr)
            lines = [ln for ln in proc.stderr.splitlines() if ln.strip()]
            if lines:
                stderr_tail = f" ({lines[-1].strip()})"
        return (
            False,
            "",
            "GPU probe crashed while testing CuPy/NVRTC"
            f"{stderr_tail}. This usually means the installed CuPy does not "
            "match the CUDA driver/toolkit on this machine. Using CPU. Set "
            "SHIFTER_DISABLE_GPU=1 to skip this probe on startup.",
        )

    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                data = json.loads(line)
                return (
                    bool(data["available"]),
                    str(data.get("name", "")),
                    str(data.get("reason", "")),
                )
            except (ValueError, KeyError):
                break

    return False, "", "GPU probe returned no usable result; using CPU"


def _detect_gpu() -> tuple[bool, str, str]:
    """Decide GPU availability, isolating the crash-prone probe by default.

    Tries two strategies in order, each in its own subprocess so a native fault
    in one cannot stop the next:

    1. ``setup_env=False`` — let CuPy use its own **bundled** CUDA libraries,
       touching neither PATH nor the DLL search dirs. This is what modern
       ``cupy-cuda12x`` / ``cupy-cuda13x`` want, and it avoids the system-toolkit ``nvrtc`` DLL
       conflict that otherwise crashes the probe.
    2. ``setup_env=True`` — inject the system CUDA toolkit path, for setups
       (e.g. conda ``cudatoolkit``) whose CuPy relies on the system libraries.

    Controlled by two environment variables:

    * ``SHIFTER_DISABLE_GPU=1`` — skip probing entirely and run on CPU.
    * ``SHIFTER_GPU_PROBE=inprocess`` — probe in-process (the old behaviour);
      faster, but a native CuPy/NVRTC fault will crash the app.
    """
    global _GPU_CUDA_VERSIONS

    if os.environ.get("SHIFTER_DISABLE_GPU") == "1":
        return False, "", "GPU disabled via SHIFTER_DISABLE_GPU=1"

    if os.environ.get("SHIFTER_GPU_PROBE", "").lower() == "inprocess":
        for strategy in _STRATEGIES:
            _GPU_CUDA_VERSIONS = _read_cuda_versions(strategy)
            available, name, reason = _probe_gpu(strategy=strategy)
            if available:
                return available, name, reason
        return available, name, reason

    last = (False, "", "")
    for strategy in _STRATEGIES:
        available, name, reason = _probe_gpu_subprocess(strategy=strategy)
        if available:
            # Reproduce the winning strategy's DLL search path in *this* process
            # for the real GPU work that runs in-process later.
            _apply_cuda_strategy(strategy)
            logger.info("GPU probe succeeded with '%s' strategy", strategy)
            return available, name, reason
        last = (available, name, reason)
    return last


def gpu_available() -> bool:
    """Return True if a suitable GPU + cupy installation is detected."""
    global _GPU_AVAILABLE, _GPU_NAME, _GPU_FAIL_REASON
    if _GPU_AVAILABLE is None:
        _GPU_AVAILABLE, _GPU_NAME, _GPU_FAIL_REASON = _detect_gpu()
        if _GPU_AVAILABLE:
            logger.info("GPU enabled: %s", _GPU_NAME)
        elif _GPU_FAIL_REASON:
            _report_gpu_unavailable(_GPU_FAIL_REASON)
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
