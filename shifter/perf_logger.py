"""Performance logging for registration and export operations.

Writes timestamped start/end markers and elapsed times to a dedicated
log file so users can audit how long each phase takes.

DEBUG-level diagnostics (the chunk-size decision, per-slab timing/throughput,
and memory-usage snapshots) are recorded **by default**. They are cheap — a
per-slab line costs microseconds against slabs that take seconds — and the log
is truncated per export, so it never accumulates across runs. To silence the
extra detail and keep only the INFO phase markers, set the ``CSC_DEBUG``
environment variable to ``0`` / ``false`` / ``no`` / ``off`` (or empty) before
launching.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import psutil

_perf_logger = logging.getLogger("shifter.perf")
_perf_logger.propagate = False  # don't duplicate to root logger

_setup_done = False
_debug_enabled = True

_TRUTHY = {"1", "true", "yes", "on"}

# Cached handle for this process so repeated memory snapshots don't re-open it.
_proc_handle: Any = None


def _resolve_debug(explicit: bool | None) -> bool:
    """Decide whether DEBUG diagnostics are on.

    An *explicit* argument wins. Otherwise DEBUG is on unless ``CSC_DEBUG`` is
    set to a falsy value — i.e. debug is the default and the env var is an
    opt-out (``CSC_DEBUG=0``), though a truthy value still explicitly enables it.
    """
    if explicit is not None:
        return bool(explicit)
    val = os.environ.get("CSC_DEBUG")
    if val is None:
        return True  # default: on
    if val.strip().lower() in _TRUTHY:
        return True
    # Any other explicit value (0, false, no, off, empty, ...) disables.
    return False


def _fmt_gib(num_bytes: float) -> str:
    """Format a byte count as a compact GiB string."""
    return f"{num_bytes / 1024**3:.2f}GiB"


def _proc() -> Any:
    global _proc_handle
    if _proc_handle is None:
        _proc_handle = psutil.Process()
    return _proc_handle


def process_rss() -> int:
    """Return this process's resident set size in bytes (0 if unavailable)."""
    try:
        return int(_proc().memory_info().rss)
    except Exception:
        return 0


def memory_status() -> str:
    """Return a compact 'process RSS / system available / % used' string."""
    vm = psutil.virtual_memory()
    rss = process_rss()
    rss_str = _fmt_gib(rss) if rss else "n/a"
    return f"proc_rss={rss_str} sys_avail={_fmt_gib(vm.available)} ({vm.percent:.0f}% used)"


def setup_perf_log(log_dir: str | Path, debug: bool | None = None) -> Path:
    """Create (or re-create) a performance log file in *log_dir*.

    Returns the path to the log file.  Subsequent calls to
    :func:`timed_operation`, :func:`log_event`, :func:`log_debug`, and
    :func:`log_memory` will write to this file.

    Parameters
    ----------
    log_dir : str | Path
        Directory to write ``performance_log.txt`` into.
    debug : bool, optional
        Force DEBUG-level diagnostics on/off.  When *None* (the default) this is
        resolved from the ``CSC_DEBUG`` environment variable, which defaults to
        **on** (set ``CSC_DEBUG=0`` to opt out).
    """
    global _setup_done, _debug_enabled
    _debug_enabled = _resolve_debug(debug)

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "performance_log.txt"

    # Remove any previous handlers so we don't duplicate.
    for h in list(_perf_logger.handlers):
        _perf_logger.removeHandler(h)
        h.close()

    handler = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d | %(levelname)-5s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    _perf_logger.addHandler(handler)
    _perf_logger.setLevel(logging.DEBUG if _debug_enabled else logging.INFO)
    _setup_done = True

    _perf_logger.info("Performance log initialised (debug=%s)", _debug_enabled)
    _perf_logger.info("CPU cores: %d", os.cpu_count() or 1)
    vm = psutil.virtual_memory()
    _perf_logger.info(
        "System RAM: total=%s available=%s",
        _fmt_gib(vm.total),
        _fmt_gib(vm.available),
    )
    return log_path


def _is_active() -> bool:
    return _setup_done and bool(_perf_logger.handlers)


def is_debug_enabled() -> bool:
    """Return True when DEBUG-level diagnostics are being recorded."""
    return _debug_enabled


def log_event(message: str) -> None:
    """Write a single informational line to the perf log."""
    if _is_active():
        _perf_logger.info(message)


def log_debug(message: str) -> None:
    """Write a DEBUG-level diagnostic line (recorded only when debug is on)."""
    if _is_active():
        _perf_logger.debug(message)


def log_memory(context: str = "", *, level: int = logging.DEBUG) -> None:
    """Log a system + process memory snapshot.

    Records this process's resident set size (RSS) and system available RAM.
    Defaults to DEBUG level; pass ``level=logging.INFO`` for snapshots that
    should be recorded even when debug is off (e.g. once at export start).
    """
    if not _is_active() or level < _perf_logger.level:
        return
    _perf_logger.log(level, "MEM | %s | %s", context or "-", memory_status())


@contextmanager
def timed_operation(operation_name: str):
    """Context manager that logs START / END timestamps and wall-clock elapsed time."""
    if not _is_active():
        yield
        return

    _perf_logger.info("START | %s", operation_name)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        _perf_logger.info(
            "END   | %s | elapsed=%.3fs", operation_name, elapsed
        )
