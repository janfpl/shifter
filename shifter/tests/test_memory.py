"""Validation for post-operation memory release (:mod:`shifter.memory`).

These tests assert *behavior* (a GC runs, GPU pools are freed only when the GPU
path ran, the heap trim is attempted and never raises) rather than the exact
number of bytes reclaimed, which is platform- and allocator-dependent.

Runnable via pytest, or standalone::

    python -m shifter.tests.test_memory
"""

from __future__ import annotations

import sys
from unittest import mock

from shifter import memory as mem
from shifter.memory import _trim_process_heap, free_gpu_memory, release_memory


def test_release_memory_returns_non_negative_int() -> None:
    r = release_memory(use_gpu=False, context="test")
    assert isinstance(r, int)
    assert r >= 0


def test_release_memory_collects_and_trims_but_skips_gpu_by_default() -> None:
    with mock.patch.object(mem.gc, "collect") as gc_collect, mock.patch.object(
        mem, "_trim_process_heap"
    ) as trim, mock.patch.object(mem, "free_gpu_memory") as free_gpu:
        release_memory(use_gpu=False)
    gc_collect.assert_called_once()
    trim.assert_called_once()
    free_gpu.assert_not_called()  # GPU pools only freed when use_gpu=True


def test_release_memory_frees_gpu_when_requested() -> None:
    with mock.patch.object(mem, "free_gpu_memory") as free_gpu, mock.patch.object(
        mem, "_trim_process_heap"
    ):
        release_memory(use_gpu=True)
    free_gpu.assert_called_once()


def test_free_gpu_memory_is_noop_without_cupy() -> None:
    # cupy is not installed in the test environment: must be a silent no-op.
    assert free_gpu_memory() is None


def test_trim_process_heap_never_raises() -> None:
    _trim_process_heap()  # must be safe on any platform


_TESTS = [
    test_release_memory_returns_non_negative_int,
    test_release_memory_collects_and_trims_but_skips_gpu_by_default,
    test_release_memory_frees_gpu_when_requested,
    test_free_gpu_memory_is_noop_without_cupy,
    test_trim_process_heap_never_raises,
]


def run_validation() -> bool:
    print("=" * 60)
    print("Memory-release Validation")
    print("=" * 60)
    all_passed = True
    for test in _TESTS:
        try:
            test()
            print(f"  PASS: {test.__name__}")
        except AssertionError as exc:
            all_passed = False
            print(f"  FAIL: {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            all_passed = False
            print(f"  ERROR: {test.__name__}: {type(exc).__name__}: {exc}")
    print("\n" + "=" * 60)
    print("OVERALL: ALL PASSED" if all_passed else "OVERALL: SOME FAILED")
    print("=" * 60)
    return all_passed


def main() -> None:
    sys.exit(0 if run_validation() else 1)


if __name__ == "__main__":
    main()
