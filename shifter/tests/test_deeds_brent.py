"""Validation for the deedsBCV (MIND-SSC, Brent) registration algorithm.

Checks known-shift recovery (including across a non-linear intensity change,
the property MIND-SSC is designed for), progress monotonicity, GPU-flag
fallback, small-volume handling, and registry wiring.

Runnable via pytest, or standalone::

    python -m shifter.tests.test_deeds_brent
"""

from __future__ import annotations

import sys

import numpy as np

from shifter.registration.deeds_brent import (
    ALGORITHM_NAME,
    DeedsBrentRegistration,
    _sample_base_coords,
)
from shifter.utils import apply_integer_shift


def _make_volume(seed: int = 0, n: int = 56) -> np.ndarray:
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


def test_deeds_brent_recovers_known_shift() -> None:
    ref = _make_volume(0)
    algo = DeedsBrentRegistration()
    for applied in [(2, -3, 1), (-4, 5, 6), (0, 0, 0)]:
        mov = apply_integer_shift(ref, applied)
        res = algo.register(ref, mov, _SR_XY, _SR_Z, use_gpu=False)
        assert (res.shift_z, res.shift_y, res.shift_x) == tuple(-a for a in applied), (
            f"applied {applied}: detected "
            f"({res.shift_z},{res.shift_y},{res.shift_x})"
        )
        assert res.algorithm_name == ALGORITHM_NAME
        assert 0.0 <= res.confidence <= 1.0


def test_deeds_brent_recovers_shift_across_intensity_change() -> None:
    ref = _make_volume(3)
    applied = (3, 2, -4)
    mov = apply_integer_shift(
        np.clip(np.sqrt(ref.astype(np.float64)) * 200.0, 0, 65535).astype(np.uint16),
        applied,
    )
    res = DeedsBrentRegistration().register(ref, mov, _SR_XY, _SR_Z, use_gpu=False)
    assert (res.shift_z, res.shift_y, res.shift_x) == tuple(-a for a in applied), (
        f"detected ({res.shift_z},{res.shift_y},{res.shift_x}), "
        f"expected {tuple(-a for a in applied)}"
    )


def test_deeds_brent_progress_is_monotonic() -> None:
    ref = _make_volume(1)
    mov = apply_integer_shift(ref, (1, -2, 3))
    fractions: list[float] = []
    DeedsBrentRegistration().register(
        ref, mov, _SR_XY, _SR_Z, use_gpu=False,
        progress_callback=lambda f: fractions.append(f),
    )
    assert fractions, "progress_callback was never called"
    assert all(0.0 <= f <= 1.0 for f in fractions)
    assert fractions == sorted(fractions), "progress went backwards"
    assert fractions[-1] == 1.0, f"progress did not reach 1.0 (last={fractions[-1]})"


def test_deeds_brent_gpu_flag_falls_back_to_cpu() -> None:
    ref = _make_volume(5)
    mov = apply_integer_shift(ref, (2, 2, -2))
    res = DeedsBrentRegistration().register(ref, mov, 5, 5, use_gpu=True)
    assert (res.shift_z, res.shift_y, res.shift_x) == (-2, -2, 2)


def test_deeds_brent_handles_small_volumes() -> None:
    """A volume with no interior sampling region still returns (keeps the seed)."""
    rng = np.random.default_rng(0)
    ref = rng.integers(0, 4000, size=(12, 14, 14)).astype(np.uint16)
    res = DeedsBrentRegistration().register(ref, ref, 3, 3, use_gpu=False)
    assert (res.shift_z, res.shift_y, res.shift_x) == (0, 0, 0)

    # _sample_base_coords returns None when the seed leaves no interior region.
    assert _sample_base_coords((20, 20, 20), (18, 0, 0), margin=3) is None


def test_deeds_brent_is_in_registry() -> None:
    from shifter.registration import ALGORITHM_REGISTRY
    from shifter.registration.base import (
        MEMORY_BYTES_PER_VOXEL,
        estimate_registration_bytes,
    )

    assert ALGORITHM_REGISTRY[ALGORITHM_NAME] is DeedsBrentRegistration
    assert ALGORITHM_NAME in MEMORY_BYTES_PER_VOXEL
    assert estimate_registration_bytes((10, 10, 10), ALGORITHM_NAME) > 0


_TESTS = [
    test_deeds_brent_recovers_known_shift,
    test_deeds_brent_recovers_shift_across_intensity_change,
    test_deeds_brent_progress_is_monotonic,
    test_deeds_brent_gpu_flag_falls_back_to_cpu,
    test_deeds_brent_handles_small_volumes,
    test_deeds_brent_is_in_registry,
]


def run_validation() -> bool:
    print("=" * 60)
    print("deedsBCV (MIND-SSC, Brent) Validation")
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
