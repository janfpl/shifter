"""Regression test: Mutual Information with an XY search range below the coarse step.

With the GUI/headless defaults (1 voxel XY, 75 voxels Z) the coarse grid used
to sample only ``-1`` in XY (``range(-1, 2, 5)``), so positive XY shifts were
unreachable, and the fine radius was ``min(5, sr_xy, sr_z) = 1`` on every axis,
so Z offsets between coarse samples could not be reached either.

Runnable via pytest, or standalone::

    python -m shifter.tests.test_mi_small_range
"""

from __future__ import annotations

import sys

import numpy as np

from shifter.registration.mutual_information import (
    _HAVE_NUMBA,
    MutualInformationRegistration,
)
from shifter.registration.mutual_information_brent import (
    MutualInformationBrentRegistration,
)
from shifter.utils import apply_integer_shift

SR_XY, SR_Z = 1, 20
# Correction shifts to recover: both XY signs, and Z offsets off the coarse grid.
CASES = [(-13, 0, 1), (12, 1, 1), (-7, -1, -1), (3, 1, -1), (18, -1, 1)]


def _volume() -> np.ndarray:
    rng = np.random.default_rng(0)
    shape = (80, 16, 60)
    vol = np.zeros(shape)
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    for _ in range(120):
        c = [rng.integers(0, s) for s in shape]
        sg = rng.uniform(2, 4)
        vol += rng.uniform(1, 5) * np.exp(
            -((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) / (2 * sg * sg)
        )
    return vol


def _check(register) -> None:
    ref = _volume()
    for true in CASES:
        mov = apply_integer_shift(ref, tuple(-t for t in true))
        r = register(ref, mov)
        got = (r.shift_z, r.shift_y, r.shift_x)
        assert got == true, f"expected {true}, got {got}"


def test_mi_serial_small_xy_range() -> None:
    algo = MutualInformationRegistration()
    _check(lambda a, b: algo._register_cpu_serial(a, b, SR_XY, SR_Z))


def test_mi_numba_small_xy_range() -> None:
    if not _HAVE_NUMBA:
        return
    algo = MutualInformationRegistration()
    _check(lambda a, b: algo._register_cpu_numba(a, b, SR_XY, SR_Z))


def test_mi_brent_small_xy_range() -> None:
    algo = MutualInformationBrentRegistration()
    _check(lambda a, b: algo.register(a, b, SR_XY, SR_Z))


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failed += 1
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if failed else 0)
