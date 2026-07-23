"""Performance logging for registration and export operations.

Writes timestamped start/end markers and elapsed times to a dedicated
log file so users can audit how long each phase takes.

Set the ``CSC_DEBUG`` environment variable (to ``1`` / ``true`` / ``yes`` /
``on``) before launching to raise the log to DEBUG level.  Debug mode adds
per-slab timing, throughput, and memory-usage diagnostics, which are useful
for tracking down slow or memory-hungry exports.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path

import psutil

_perf_logger = logging.getLogger("shifter.perf")
_perf_logger.propagate = False  # don't duplicate to root logger

_setup_done = False
_debug_enabled = False

_TRUTHY = {"1", "true", "yes", "on"}


def _env_debug() -> bool:
    """Return True if the ``CSC_DEBUG`` environment variable is truthy."""
    return os.environ.get("CSC_DEBUG", "").strip().lower() in _TRUTHY


def _fmt_gib(num_bytes: float) -> str:
    """Format a byte count as a compact GiB string."""
    return f"{num_bytes / 1024**3:.2f}GiB"


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
        Enable DEBUG-level diagnostics (per-slab timing/throughput and memory
        snapshots).  When *None* (the default) this is taken from the
        ``CSC_DEBUG`` environment variable.
    """
    global _setup_done, _debug_enabled
    _debug_enabled = _env_debug() if debug is None else bool(debug)

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

    Records total / used / available system RAM and this process's resident
    set size (RSS).  Defaults to DEBUG level (suppressed unless debug logging
    is on); pass ``level=logging.INFO`` for snapshots that should always be
    recorded (e.g. once at export start).
    """
    if not _is_active() or level < _perf_logger.level:
        return
    vm = psutil.virtual_memory()
    try:
        rss = _fmt_gib(psutil.Process().memory_info().rss)
    except Exception:
        rss = "n/a"
    _perf_logger.log(
        level,
        "MEM | %s | proc_rss=%s sys_used=%s/%s (%.0f%%) sys_avail=%s",
        context or "-",
        rss,
        _fmt_gib(vm.total - vm.available),
        _fmt_gib(vm.total),
        vm.percent,
        _fmt_gib(vm.available),
    )


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
