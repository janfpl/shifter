"""Validation for granular registration progress reporting.

Focuses on the mutual-information algorithm (the slow, searching one that most
needs sub-step progress) and asserts two things:

- **Behavior preservation:** adding a ``progress_callback`` — which internally
  batches the grid search — does not change the computed result.
- **Granularity:** the callback receives a monotonically non-decreasing stream
  of fractions with several distinct values, ending at 1.0.

Also checks that every registered algorithm's ``register`` accepts the
``progress_callback`` keyword (the worker calls them all uniformly).

Uses only numpy for the MI path, so it runs without scipy/scikit-image.

Runnable via pytest, or standalone::

    python -m shifter.tests.test_registration_progress
"""

from __future__ import annotations

import sys

import numpy as np

from shifter.registration.mutual_information import MutualInformationRegistration
from shifter.utils import apply_integer_shift


def _make_volume(seed: int = 0) -> np.ndarray:
    """A small volume with a few sharp Gaussian blobs (enough structure for MI)."""
    rng = np.random.default_rng(seed)
    nz = ny = nx = 32
    vol = np.zeros((nz, ny, nx), dtype=np.float64)
    zz, yy, xx = np.ogrid[:nz, :ny, :nx]
    for _ in range(8):
        cz, cy, cx = rng.integers(8, 24, size=3)
        sigma = rng.uniform(2.0, 4.0)
        vol += rng.uniform(5000, 40000) * np.exp(
            -((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2)
        )
    return np.clip(vol, 0, 65535).astype(np.uint16)


_SR_XY = 5
_SR_Z = 4


def test_mi_progress_callback_does_not_change_result() -> None:
    """A progress_callback must not alter the detected shift or metric."""
    ref = _make_volume(0)
    mov = apply_integer_shift(ref, (2, -3, 1))  # applied shift
    algo = MutualInformationRegistration()

    without = algo.register(ref, mov, _SR_XY, _SR_Z, use_gpu=False)
    with_cb = algo.register(
        ref, mov, _SR_XY, _SR_Z, use_gpu=False, progress_callback=lambda f: None
    )

    assert (without.shift_x, without.shift_y, without.shift_z) == (
        with_cb.shift_x, with_cb.shift_y, with_cb.shift_z
    )
    assert without.raw_metric_value == with_cb.raw_metric_value


def test_mi_progress_is_monotonic_and_granular() -> None:
    """The callback stream is non-decreasing, ends at 1.0, and is fine-grained."""
    ref = _make_volume(1)
    mov = apply_integer_shift(ref, (1, 2, -2))
    algo = MutualInformationRegistration()

    fractions: list[float] = []
    algo.register(
        ref, mov, _SR_XY, _SR_Z, use_gpu=False,
        progress_callback=lambda f: fractions.append(f),
    )

    assert fractions, "progress_callback was never called"
    assert all(0.0 <= f <= 1.0 for f in fractions)
    assert fractions == sorted(fractions), "progress went backwards"
    assert fractions[-1] == 1.0, f"progress did not reach 1.0 (last={fractions[-1]})"
    assert len(set(fractions)) > 2, "progress was not granular (<=2 distinct values)"


def test_mi_recovers_known_shift() -> None:
    """Sanity: MI still detects the applied shift's correction (negated)."""
    ref = _make_volume(2)
    applied = (2, -3, 1)
    mov = apply_integer_shift(ref, applied)
    algo = MutualInformationRegistration()

    res = algo.register(ref, mov, _SR_XY, _SR_Z, use_gpu=False)
    assert (res.shift_z, res.shift_y, res.shift_x) == (
        -applied[0], -applied[1], -applied[2]
    ), f"detected ({res.shift_z},{res.shift_y},{res.shift_x}), expected {tuple(-a for a in applied)}"


def test_all_algorithms_accept_progress_callback() -> None:
    """Every registered algorithm's register() must accept progress_callback."""
    import inspect

    from shifter.registration import ALGORITHM_REGISTRY

    for name, cls in ALGORITHM_REGISTRY.items():
        params = inspect.signature(cls.register).parameters
        assert "progress_callback" in params, f"{name} lacks progress_callback"


_TESTS = [
    test_mi_progress_callback_does_not_change_result,
    test_mi_progress_is_monotonic_and_granular,
    test_mi_recovers_known_shift,
    test_all_algorithms_accept_progress_callback,
]


def run_validation() -> bool:
    print("=" * 60)
    print("Registration progress Validation")
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
