"""Validation for memory-safe export chunking and slab reads.

Covers the fix for the full-volume export out-of-memory crash, where
``compute_chunk_size`` sized a single Z-slab at tens of GiB (a reported case
allocated a ``(828, 6979, 5347)`` uint16 slab = 57.6 GiB on a machine that was
already >90% full):

- :func:`compute_chunk_size` budgets against *available* RAM (not total), never
  exceeds the absolute slab-size cap, and always returns >= 1.
- :func:`_read_full_slab` reproduces the reference zero-padded integer Z-shift
  exactly, while returning the dask result directly (no extra full-size copy)
  for interior slabs, and never corrupts the source array when the slab is
  mutated in place by the subsequent XY shift.

Runnable via pytest, or standalone::

    python -m chromatic_shift_corrector.tests.test_export_memory
"""

from __future__ import annotations

import sys
from unittest import mock

import dask.array as da
import numpy as np

from chromatic_shift_corrector import export_engine as ee
from chromatic_shift_corrector.export_engine import (
    _MAX_SLAB_BYTES,
    _SLAB_PEAK_COPIES,
    _read_full_slab,
    compute_chunk_size,
)
from chromatic_shift_corrector.utils import apply_integer_shift


class _FakeVM:
    """Minimal stand-in for ``psutil.virtual_memory()``'s return value."""

    def __init__(self, total: int, available: int) -> None:
        self.total = total
        self.available = available
        self.used = total - available
        self.percent = 100.0 * (total - available) / total


def _patch_ram(total_gib: float, avail_gib: float):
    vm = _FakeVM(int(total_gib * 1024**3), int(avail_gib * 1024**3))
    return mock.patch.object(ee.psutil, "virtual_memory", return_value=vm)


# ------------------------------------------------------------------------- #
# compute_chunk_size
# ------------------------------------------------------------------------- #

def test_chunk_size_never_exceeds_cap() -> None:
    """Even with vast free RAM, one slab must stay within the absolute cap."""
    ny, nx = 6979, 5347  # the plane size from the crash report
    plane = ny * nx * 2
    with _patch_ram(total_gib=1024, avail_gib=1000):
        cz = compute_chunk_size((ny, nx), n_channels=1, ram_percent=90)
    assert cz >= 1
    assert cz * plane <= _MAX_SLAB_BYTES, "slab exceeds the hard cap"


def test_chunk_size_regression_reported_case() -> None:
    """The reported 128 GiB machine must not reproduce the 828-plane / 57 GiB slab."""
    ny, nx = 6979, 5347
    plane = ny * nx * 2
    # 128 GiB box, fully free: the old model produced chunk_z=828 (57.6 GiB).
    with _patch_ram(total_gib=128, avail_gib=128):
        cz = compute_chunk_size((ny, nx), n_channels=1, ram_percent=90)
    slab_gib = cz * plane / 1024**3
    assert cz < 828, f"regressed to an oversized chunk_z={cz}"
    assert slab_gib <= _MAX_SLAB_BYTES / 1024**3 + 1e-9
    assert slab_gib <= 4.0, f"slab still large: {slab_gib:.1f} GiB"


def test_chunk_size_budgets_available_not_total() -> None:
    """A machine that is >90% full must size slabs against the small free pool."""
    ny, nx = 4096, 4096
    plane = ny * nx * 2
    # 128 GiB total, but only 6 GiB actually available (system already loaded).
    with _patch_ram(total_gib=128, avail_gib=6):
        cz = compute_chunk_size((ny, nx), n_channels=1, ram_percent=90)
    budget = int(6 * 1024**3 * 0.90)
    peak = cz * plane * _SLAB_PEAK_COPIES
    assert cz >= 1
    assert peak <= budget, "estimated peak exceeds the available-RAM budget"


def test_chunk_size_minimum_one() -> None:
    """A single plane larger than the whole budget still yields chunk_z >= 1."""
    ny, nx = 20000, 20000  # ~0.75 GiB per plane
    with _patch_ram(total_gib=1, avail_gib=0.2):
        cz = compute_chunk_size((ny, nx), n_channels=1, ram_percent=90)
    assert cz == 1


def test_chunk_size_lower_ram_percent_shrinks() -> None:
    """When RAM (not the cap) binds, a lower allocation yields a smaller slab."""
    ny, nx = 2048, 2048
    # avail small enough that the RAM budget (< 3x cap) is the binding limit.
    with _patch_ram(total_gib=32, avail_gib=8):
        hi = compute_chunk_size((ny, nx), n_channels=1, ram_percent=90)
        lo = compute_chunk_size((ny, nx), n_channels=1, ram_percent=50)
    assert lo < hi, f"expected lower RAM%% to shrink the slab (lo={lo}, hi={hi})"


# ------------------------------------------------------------------------- #
# _read_full_slab
# ------------------------------------------------------------------------- #

def _reassemble_via_slabs(vol: np.ndarray, sz: int, chunk_z: int) -> np.ndarray:
    """Rebuild a Z-shifted volume slab-by-slab, as the export loop does."""
    nz, ny, nx = vol.shape
    d = da.from_array(vol, chunks=(min(chunk_z, nz), -1, -1))
    out = np.empty((nz, ny, nx), dtype=np.uint16)
    for z0 in range(0, nz, chunk_z):
        z1 = min(z0 + chunk_z, nz)
        out[z0:z1] = _read_full_slab(d, z0, z1, sz)
    return out


def test_read_full_slab_matches_reference_shift() -> None:
    """_read_full_slab reproduces apply_integer_shift on Z for all shifts/chunkings."""
    rng = np.random.default_rng(0)
    vol = rng.integers(0, 65535, size=(37, 16, 12), dtype=np.uint16)
    nz = vol.shape[0]
    for sz in [-nz - 3, -8, -1, 0, 1, 8, nz + 3]:
        expected = apply_integer_shift(vol, (sz, 0, 0))
        for chunk_z in [1, 5, 8, 37, 50]:
            got = _reassemble_via_slabs(vol, sz, chunk_z)
            assert np.array_equal(got, expected), f"sz={sz} chunk_z={chunk_z}"


def test_read_full_slab_interior_skips_zero_alloc() -> None:
    """An interior slab (no Z-pad) must not allocate a second zero-filled buffer."""
    vol = np.arange(20 * 4 * 4, dtype=np.uint16).reshape(20, 4, 4)
    d = da.from_array(vol, chunks=(8, -1, -1))
    with mock.patch.object(ee.np, "zeros", wraps=np.zeros) as mz:
        slab = _read_full_slab(d, 4, 12, sz=0)  # fully interior, no shift
    assert np.array_equal(slab, vol[4:12])
    mz.assert_not_called()


def test_read_full_slab_boundary_pads() -> None:
    """A boundary slab zero-pads the vacated planes and copies the rest."""
    vol = np.ones((10, 4, 4), dtype=np.uint16)
    d = da.from_array(vol, chunks=(4, -1, -1))
    slab = _read_full_slab(d, 0, 5, sz=2)  # first 2 output planes vacated
    assert slab.shape == (5, 4, 4)
    assert np.array_equal(slab[:2], np.zeros((2, 4, 4), dtype=np.uint16))
    assert np.array_equal(slab[2:], vol[0:3])


def test_read_full_slab_mutation_does_not_corrupt_source() -> None:
    """Mutating the returned slab must never write back into the source array."""
    vol = np.ones((12, 4, 4), dtype=np.uint16)
    src_before = vol.copy()
    d = da.from_array(vol, chunks=(4, -1, -1))
    slab = _read_full_slab(d, 0, 8, sz=0)
    slab[...] = 42  # emulate the in-place XY shift
    assert np.array_equal(vol, src_before), "source corrupted by slab mutation"


def _export_full_via_loop(
    vol: np.ndarray, sz: int, sy: int, sx: int, chunk_z: int
) -> np.ndarray:
    """Replicate export_channel's full-volume slab loop (read + in-place XY shift)."""
    nz, ny, nx = vol.shape
    d = da.from_array(vol, chunks=(min(chunk_z, nz), -1, -1))
    out = np.empty((nz, ny, nx), dtype=np.uint16)
    for z0 in range(0, nz, chunk_z):
        z1 = min(z0 + chunk_z, nz)
        slab = _read_full_slab(d, z0, z1, sz)
        slab = ee._shift_slab_xy(slab, sy, sx)
        out[z0:z1] = slab
    return out


def test_full_volume_loop_matches_reference_3d_shift() -> None:
    """The refactored full-volume path exactly reproduces a 3D integer shift.

    This also guards against slab aliasing: if a returned slab shared memory
    with the source volume, the in-place XY shift would corrupt planes not yet
    read and the comparison would fail.
    """
    rng = np.random.default_rng(1)
    vol = rng.integers(0, 65535, size=(29, 20, 24), dtype=np.uint16)
    for sz, sy, sx in [(0, 0, 0), (3, -2, 5), (-4, 6, -7), (1, 0, -3), (0, 2, 0)]:
        expected = apply_integer_shift(vol, (sz, sy, sx))
        for chunk_z in [1, 4, 7, 29]:
            got = _export_full_via_loop(vol, sz, sy, sx, chunk_z)
            assert np.array_equal(got, expected), (
                f"shift={(sz, sy, sx)} chunk_z={chunk_z}"
            )


# ------------------------------------------------------------------------- #
# Standalone runner
# ------------------------------------------------------------------------- #

_TESTS = [
    test_chunk_size_never_exceeds_cap,
    test_chunk_size_regression_reported_case,
    test_chunk_size_budgets_available_not_total,
    test_chunk_size_minimum_one,
    test_chunk_size_lower_ram_percent_shrinks,
    test_read_full_slab_matches_reference_shift,
    test_read_full_slab_interior_skips_zero_alloc,
    test_read_full_slab_boundary_pads,
    test_read_full_slab_mutation_does_not_corrupt_source,
    test_full_volume_loop_matches_reference_3d_shift,
]


def run_validation() -> bool:
    print("=" * 60)
    print("Export memory-safety Validation")
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
