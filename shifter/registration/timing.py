"""Lightweight phase timing for registration algorithms.

Registration wall-clock time is dominated by a few phases (coarse grid search,
fine/Brent refinement, descriptor build, per-pyramid-level cost). This records
per-phase timings so it's clear *where* a registration spends its time on real
data — useful when one method is unexpectedly slower than another.

Each timing line is emitted twice:

* through the module logger (``shifter.registration.timing``), which propagates
  to the root logger and therefore the terminal (see ``shifter.logging_setup``);
* through :func:`shifter.perf_logger.log_event`, so it also lands in
  ``performance_log.txt`` when a perf log is active (during a registration run).

Lines are prefixed ``[timing]`` so they are easy to grep.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager

_logger = logging.getLogger("shifter.registration.timing")


def _emit(message: str) -> None:
    _logger.info("[timing] %s", message)
    try:
        from shifter.perf_logger import log_event

        log_event(f"[timing] {message}")
    except Exception:
        pass


def note(message: str) -> None:
    """Emit a one-off timing/diagnostic line (e.g. an evaluation count)."""
    _emit(message)


@contextmanager
def phase(name: str):
    """Time a named phase and emit its elapsed wall-clock seconds on exit."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _emit(f"{name}: {time.perf_counter() - t0:.2f}s")
