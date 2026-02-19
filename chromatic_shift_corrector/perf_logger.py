"""Performance logging for registration and export operations.

Writes timestamped start/end markers and elapsed times to a dedicated
log file so users can audit how long each phase takes.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path

_perf_logger = logging.getLogger("chromatic_shift_corrector.perf")
_perf_logger.propagate = False  # don't duplicate to root logger

_setup_done = False


def setup_perf_log(log_dir: str | Path) -> Path:
    """Create (or re-create) a performance log file in *log_dir*.

    Returns the path to the log file.  Subsequent calls to
    :func:`timed_operation` will write to this file.
    """
    global _setup_done
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
            "%(asctime)s.%(msecs)03d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    _perf_logger.addHandler(handler)
    _perf_logger.setLevel(logging.INFO)
    _setup_done = True

    _perf_logger.info("Performance log initialised")
    _perf_logger.info("CPU cores: %d", os.cpu_count() or 1)
    return log_path


def _is_active() -> bool:
    return _setup_done and _perf_logger.handlers


def log_event(message: str) -> None:
    """Write a single informational line to the perf log."""
    if _is_active():
        _perf_logger.info(message)


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
