"""Validation for the deedsBCV (MIND-SSC) registration algorithm.

Checks the descriptor itself (shape, non-negativity, invariance to an affine
intensity change — the property that makes MIND-SSC useful across channels)
and the coarse-to-fine translation search (known-shift recovery, progress
reporting, GPU-flag fallback).

Runnable via pytest, or standalone::

    python -m shifter.tests.test_deeds
"""

from __future__ import annotations

import sys

import numpy as np

from shifter.registration.deeds import (
    ALGORITHM_NAME,
    DeedsRegistration,
    _pyramid_factors,
    mind_ssc,
)
from shifter.utils import apply_integer_shift


def _make_volume(seed: int = 0, n: int = 48) -> np.ndarray:
    """A volume with enough blob structure for a descriptor-based search."""
    rng = np.random.default_rng(seed)
    vol = np.zeros((n, n, n), dtype=np.float64)
    zz, yy, xx = np.ogrid[:n, :n, :n]
    for _ in range(12):
        cz, cy, cx = rng.integers(10, n - 10, size=3)
        sigma = rng.uniform(2.0, 4.5)
        vol += rng.uniform(5000, 40000) * np.exp(
            -((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2)
        )
    return np.clip(vol, 0, 65535).astype(np.uint16)


_SR_XY = 8
_SR_Z = 8


def test_mind_ssc_shape_and_range() -> None:
    """The descriptor has 12 channels per voxel, all in (0, 1]."""
    desc = mind_ssc(_make_volume(0, n=24))

    assert desc.shape == (12,) + (24, 24, 24), f"unexpected shape {desc.shape}"
    assert desc.dtype == np.float32
    assert np.all(desc > 0.0) and np.all(desc <= 1.0 + 1e-6)
    # The descriptor is min-subtracted before exp(-x), so each voxel has at
    # least one entry equal to 1.
    assert np.allclose(desc.max(axis=0), 1.0, atol=1e-5)


def test_mind_ssc_is_intensity_invariant() -> None:
    """An affine intensity change leaves the descriptor unchanged.

    This is the property that lets the algorithm register channels with
    unrelated brightness/contrast.
    """
    vol = _make_volume(1, n=24).astype(np.float64)
    base = mind_ssc(vol)
    scaled = mind_ssc(vol * 3.0 + 1000.0)

    assert np.allclose(base, scaled, atol=1e-4), (
        f"max deviation {np.abs(base - scaled).max():.6f}"
    )


def test_deeds_recovers_known_shift() -> None:
    """The detected shift is the correction (negated applied shift)."""
    ref = _make_volume(2)
    for applied in [(2, -3, 1), (-4, 5, 6), (0, 0, 0)]:
        mov = apply_integer_shift(ref, applied)
        res = DeedsRegistration().register(ref, mov, _SR_XY, _SR_Z, use_gpu=False)

        assert (res.shift_z, res.shift_y, res.shift_x) == tuple(-a for a in applied), (
            f"applied {applied}: detected "
            f"({res.shift_z},{res.shift_y},{res.shift_x})"
        )
        assert res.algorithm_name == ALGORITHM_NAME
        assert 0.0 <= res.confidence <= 1.0


def test_deeds_recovers_shift_across_intensity_change() -> None:
    """A channel with a different intensity response still registers."""
    ref = _make_volume(3)
    applied = (3, 2, -4)
    mov = apply_integer_shift(
        np.clip(np.sqrt(ref.astype(np.float64)) * 200.0, 0, 65535).astype(np.uint16),
        applied,
    )

    res = DeedsRegistration().register(ref, mov, _SR_XY, _SR_Z, use_gpu=False)
    assert (res.shift_z, res.shift_y, res.shift_x) == tuple(-a for a in applied), (
        f"detected ({res.shift_z},{res.shift_y},{res.shift_x}), "
        f"expected {tuple(-a for a in applied)}"
    )


def test_deeds_progress_callback_is_monotonic_and_harmless() -> None:
    """Progress is non-decreasing, ends at 1.0, and does not change the result."""
    ref = _make_volume(4)
    mov = apply_integer_shift(ref, (1, -2, 3))
    algo = DeedsRegistration()

    without = algo.register(ref, mov, _SR_XY, _SR_Z, use_gpu=False)
    fractions: list[float] = []
    with_cb = algo.register(
        ref, mov, _SR_XY, _SR_Z, use_gpu=False,
        progress_callback=lambda f: fractions.append(f),
    )

    assert (without.shift_x, without.shift_y, without.shift_z) == (
        with_cb.shift_x, with_cb.shift_y, with_cb.shift_z
    )
    assert without.raw_metric_value == with_cb.raw_metric_value
    assert fractions, "progress_callback was never called"
    assert all(0.0 <= f <= 1.0 for f in fractions)
    assert fractions == sorted(fractions), "progress went backwards"
    assert fractions[-1] == 1.0, f"progress did not reach 1.0 (last={fractions[-1]})"
    assert len(set(fractions)) > 2, "progress was not granular"


def test_deeds_gpu_flag_falls_back_to_cpu() -> None:
    """``use_gpu=True`` without cupy must fall back, not raise."""
    ref = _make_volume(5, n=32)
    mov = apply_integer_shift(ref, (2, 2, -2))

    res = DeedsRegistration().register(ref, mov, 5, 5, use_gpu=True)
    assert (res.shift_z, res.shift_y, res.shift_x) == (-2, -2, 2)


def test_deeds_handles_small_volumes() -> None:
    """Volumes too small to downsample still register (single-level pyramid)."""
    assert _pyramid_factors((8, 10, 12)) == [1]

    rng = np.random.default_rng(0)
    ref = rng.integers(0, 4000, size=(10, 14, 14)).astype(np.uint16)
    res = DeedsRegistration().register(ref, ref, 3, 3, use_gpu=False)
    assert (res.shift_z, res.shift_y, res.shift_x) == (0, 0, 0)


def test_deeds_is_in_registry() -> None:
    """The widget's algorithm dropdown is built from the registry."""
    from shifter.registration import ALGORITHM_REGISTRY
    from shifter.registration.base import (
        MEMORY_BYTES_PER_VOXEL,
        estimate_registration_bytes,
    )

    assert ALGORITHM_REGISTRY[ALGORITHM_NAME] is DeedsRegistration
    assert ALGORITHM_NAME in MEMORY_BYTES_PER_VOXEL
    assert estimate_registration_bytes((10, 10, 10), ALGORITHM_NAME) > 0


_TESTS = [
    test_mind_ssc_shape_and_range,
    test_mind_ssc_is_intensity_invariant,
    test_deeds_recovers_known_shift,
    test_deeds_recovers_shift_across_intensity_change,
    test_deeds_progress_callback_is_monotonic_and_harmless,
    test_deeds_gpu_flag_falls_back_to_cpu,
    test_deeds_handles_small_volumes,
    test_deeds_is_in_registry,
]


def run_validation() -> bool:
    print("=" * 60)
    print("deedsBCV (MIND-SSC) Validation")
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
