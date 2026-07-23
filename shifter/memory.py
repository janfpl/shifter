"""Best-effort release of RAM held after a large operation.

Python's allocator and the C runtime heap hold on to freed memory so they can
reuse it, so a process's resident set size stays near its high-water mark even
after the big arrays are dropped. Registration is a notable offender: the
mutual-information / FFT paths build several ``float64`` copies of the
sub-volume, and once registration finishes those copies are unreferenced but
still counted against the process.

That retained memory is not "leaked" — the same process can reuse it — but it
has a real downstream cost here: :func:`shifter.export_engine.compute_chunk_size`
sizes export slabs from ``psutil.virtual_memory().available``, and memory the
allocator is hoarding does *not* count as available, so a bloated post-
registration footprint makes the next export pick smaller slabs than it could.

This module drops that memory back to the OS as far as the platform allows:

- a garbage collection to break any reference cycles;
- freeing CuPy's device **and** pinned-host memory pools (the pinned pool is
  system RAM) when the GPU path was used;
- an advisory C-runtime heap trim (glibc ``malloc_trim`` / Windows CRT
  ``_heapmin``) to return freed pages to the OS.

Effectiveness of the heap trim varies by platform and allocation pattern, so
:func:`release_memory` logs a before/after snapshot: the performance log shows
exactly how much came back on the machine it ran on.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import gc
import logging
import platform

from shifter.perf_logger import log_event, log_memory, process_rss

logger = logging.getLogger(__name__)


def free_gpu_memory() -> None:
    """Return CuPy's cached device + pinned-host memory to the driver / OS.

    No-op when cupy isn't installed or no GPU work has run.
    """
    try:
        import cupy as cp
    except Exception:
        return
    for pool_getter in ("get_default_memory_pool", "get_default_pinned_memory_pool"):
        try:
            getattr(cp, pool_getter)().free_all_blocks()
        except Exception:
            pass


def _trim_process_heap() -> None:
    """Advise the C runtime to return freed heap pages to the OS.

    glibc exposes ``malloc_trim``; the Windows CRT exposes ``_heapmin``. Both
    are advisory and safe to call; anything unexpected is swallowed so this can
    never break the caller.
    """
    system = platform.system()
    try:
        if system == "Linux":
            libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
            if hasattr(libc, "malloc_trim"):
                libc.malloc_trim(0)
        elif system == "Windows":
            # _heapmin() releases unused CRT heap pages back to the OS.
            ctypes.cdll.msvcrt._heapmin()
    except Exception:
        pass


def release_memory(use_gpu: bool = False, *, context: str = "") -> int:
    """Reclaim RAM held after a large operation; return bytes returned to the OS.

    Runs a garbage collection, frees CuPy pools (when *use_gpu*), and asks the C
    runtime to return freed heap to the OS. Logs a before/after memory snapshot
    to the performance log (at INFO, so it is recorded even with debug off) so
    the effect is visible.

    Parameters
    ----------
    use_gpu : bool
        Whether the preceding operation used the GPU (frees CuPy pools if so).
    context : str
        Short label for the log lines (e.g. ``"registration"``).

    Returns
    -------
    int
        Approximate bytes reclaimed (drop in process RSS; 0 if none / unknown).
    """
    label = context or "release"
    rss_before = process_rss()
    log_memory(f"{label}: before cleanup", level=logging.INFO)

    gc.collect()
    if use_gpu:
        free_gpu_memory()
    _trim_process_heap()

    rss_after = process_rss()
    reclaimed = max(0, rss_before - rss_after)
    log_memory(f"{label}: after cleanup", level=logging.INFO)
    log_event(
        f"Memory cleanup ({label}): reclaimed {reclaimed / 1024**3:.2f} GiB "
        f"(RSS {rss_before / 1024**3:.2f} -> {rss_after / 1024**3:.2f} GiB)"
    )
    logger.info(
        "Memory cleanup (%s): reclaimed %.2f GiB (RSS %.2f -> %.2f GiB)",
        label, reclaimed / 1024**3, rss_before / 1024**3, rss_after / 1024**3,
    )
    return reclaimed
