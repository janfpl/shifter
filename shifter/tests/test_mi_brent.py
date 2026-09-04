"""Validation for the Mutual Information (Brent) registration algorithm.

Checks known-shift recovery (including across a non-linear intensity change,
the mutual-information use case), monotonic progress reporting, registry
wiring, and that it converges faster than the exhaustive grid MI.

Runnable via pytest, or standalone::

    python -m shifter.tests.test_mi_brent
"""

from __future__ import annotations

import sys
import time

import numpy as np

from shifter.registration.mutual_information_brent import (
    ALGORITHM_NAME,
    MutualInformationBrentRegistration,
    _neg_mi_continuous,
)
from shifter.utils import apply_integer_shift


def _make_volume(seed: int = 0, n: int = 48) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vol = np.zeros((n, n, n), dtype=np.float64)
    zz, yy, xx = np.ogrid[:n, :n, :n]
    for _ in range(12):
        cz, cy, cx = rng.integers(9, n - 9, size=3)
        sigma = rng.uniform(2.0, 4.5)
        vol += rng.uniform(5000, 40000) * np.exp(
            -((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2)
        )
    return np.clip(vol, 0, 65535).astype(np.uint16)


_SR_XY = 8
_SR_Z = 8


def test_brent_recovers_known_shift() -> None:
    ref = _make_volume(0)
    algo = MutualInformationBrentRegistration()
    for applied in [(2, -3, 1), (-4, 5, 6), (0, 0, 0)]:
        mov = apply_integer_shift(ref, applied)
        res = algo.register(ref, mov, _SR_XY, _SR_Z, use_gpu=False)
        assert (res.shift_z, res.shift_y, res.shift_x) == tuple(-a for a in applied), (
            f"applied {applied}: detected "
            f"({res.shift_z},{res.shift_y},{res.shift_x})"
        )
        assert res.algorithm_name == ALGORITHM_NAME
        assert 0.0 <= res.confidence <= 1.0


def test_brent_recovers_shift_across_intensity_change() -> None:
    """A channel with a different intensity response still registers."""
    ref = _make_volume(3)
    applied = (3, 2, -4)
    mov = apply_integer_shift(
        np.clip(np.sqrt(ref.astype(np.float64)) * 200.0, 0, 65535).astype(np.uint16),
        applied,
    )
    res = MutualInformationBrentRegistration().register(
        ref, mov, _SR_XY, _SR_Z, use_gpu=False
    )
    assert (res.shift_z, res.shift_y, res.shift_x) == tuple(-a for a in applied), (
        f"detected ({res.shift_z},{res.shift_y},{res.shift_x}), "
        f"expected {tuple(-a for a in applied)}"
    )


def test_brent_progress_is_monotonic() -> None:
    ref = _make_volume(1)
    mov = apply_integer_shift(ref, (1, -2, 3))
    fractions: list[float] = []
    MutualInformationBrentRegistration().register(
        ref, mov, _SR_XY, _SR_Z, use_gpu=False,
        progress_callback=lambda f: fractions.append(f),
    )
    assert fractions, "progress_callback was never called"
    assert all(0.0 <= f <= 1.0 for f in fractions)
    assert fractions == sorted(fractions), "progress went backwards"
    assert fractions[-1] == 1.0, f"progress did not reach 1.0 (last={fractions[-1]})"


def test_neg_mi_objective_minimised_at_true_shift() -> None:
    """-MI is (near-)lowest at the true shift and higher one voxel away."""
    ref = _make_volume(2)
    applied = (2, -1, 3)
    mov = apply_integer_shift(ref, applied)
    correction = tuple(-a for a in applied)  # (dz, dy, dx)

    at_truth = _neg_mi_continuous(ref.astype(np.float64), mov.astype(np.float64),
                                  *correction)
    for axis in range(3):
        off = list(correction)
        off[axis] += 2
        assert _neg_mi_continuous(
            ref.astype(np.float64), mov.astype(np.float64), *off
        ) > at_truth, f"objective not higher when axis {axis} is off by 2"


def test_brent_is_faster_than_grid_mi() -> None:
    """Brent refinement should beat the exhaustive fine grid on wall-clock."""
    from shifter.registration import ALGORITHM_REGISTRY

    ref = _make_volume(4, n=64)
    mov = apply_integer_shift(ref, (3, -4, 5))

    brent = ALGORITHM_REGISTRY[ALGORITHM_NAME]()
    grid = ALGORITHM_REGISTRY["Mutual Information"]()

    # Warm up numba so the JIT compile is not charged to either timing.
    brent.register(ref, mov, 4, 4, use_gpu=False)
    grid.register(ref, mov, 4, 4, use_gpu=False)

    t = time.time(); rb = brent.register(ref, mov, 10, 10, use_gpu=False); tb = time.time() - t
    t = time.time(); rg = grid.register(ref, mov, 10, 10, use_gpu=False); tg = time.time() - t

    assert (rb.shift_z, rb.shift_y, rb.shift_x) == (rg.shift_z, rg.shift_y, rg.shift_x), (
        "Brent and grid MI disagreed on the shift"
    )
    assert tb < tg, f"Brent ({tb:.2f}s) was not faster than grid MI ({tg:.2f}s)"


def test_brent_is_in_registry() -> None:
    from shifter.registration import ALGORITHM_REGISTRY
    from shifter.registration.base import (
        MEMORY_BYTES_PER_VOXEL,
        estimate_registration_bytes,
    )

    assert ALGORITHM_REGISTRY[ALGORITHM_NAME] is MutualInformationBrentRegistration
    assert ALGORITHM_NAME in MEMORY_BYTES_PER_VOXEL
    assert estimate_registration_bytes((10, 10, 10), ALGORITHM_NAME) > 0


_TESTS = [
    test_brent_recovers_known_shift,
    test_brent_recovers_shift_across_intensity_change,
    test_brent_progress_is_monotonic,
    test_neg_mi_objective_minimised_at_true_shift,
    test_brent_is_faster_than_grid_mi,
    test_brent_is_in_registry,
]


def run_validation() -> bool:
    print("=" * 60)
    print("Mutual Information (Brent) Validation")
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
