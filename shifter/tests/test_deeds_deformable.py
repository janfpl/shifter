"""Validation for the deformable deedsBCV registration.

Covers the primitives (message DT vs brute force, MST validity, field warp and
inverse-consistent composition) and the end-to-end solver (known-translation and
smooth-deformation recovery), plus a regression guard that the integer pipeline
is untouched. Most end-to-end tests use a reduced level schedule to stay fast; a
faithful full-schedule run is included as an opt-in slow check.

Runnable via pytest, or standalone::

    python -m shifter.tests.test_deeds_deformable
"""

from __future__ import annotations

import sys

import numpy as np

from shifter.registration.deeds_field import (
    warp_volume,
    upsample_field,
    compose_consistent,
    _compose,
)
from shifter.registration.deeds_mst import (
    message_dt_batch,
    prims_graph,
)
from shifter.registration.deeds_deformable import (
    ALGORITHM_NAME,
    DeformableResult,
    register_deformable,
    warp_corrected,
)
from shifter.utils import apply_integer_shift


# Reduced schedule: (grid_spacing, search_radius, quantisation, mind_step).
_FAST = [(8, 4, 2, 2), (4, 3, 1, 1)]


def _texture(seed: int = 0, n: int = 48) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vol = np.zeros((n, n, n), dtype=np.float64)
    zz, yy, xx = np.ogrid[:n, :n, :n]
    for _ in range(20):
        cz, cy, cx = rng.integers(6, n - 6, size=3)
        sigma = rng.uniform(2.0, 4.0)
        vol += rng.uniform(5000, 40000) * np.exp(
            -((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2)
        )
    return np.clip(vol, 0, 65535).astype(np.uint16)


def _sinusoid_field(n: int, amp: float = 2.5) -> np.ndarray:
    zz, yy, xx = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    return np.stack(
        [
            amp * np.sin(2 * np.pi * yy / n),
            amp * np.sin(2 * np.pi * xx / n),
            amp * np.sin(2 * np.pi * zz / n),
        ]
    ).astype(np.float32)


def _ssd(a, b):
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


# --------------------------------------------------------------------------- #
# Primitive unit tests
# --------------------------------------------------------------------------- #

def test_message_dt_matches_brute_force() -> None:
    rng = np.random.default_rng(0)
    L, N = 5, 2
    cost = (rng.random((N, L, L, L)) * 10).astype(np.float32)
    offs = ((rng.random((N, 3)) - 0.5) * 2).astype(np.float32)
    msg, src = message_dt_batch(cost, offs)
    for n in range(N):
        for o0 in range(L):
            for o1 in range(L):
                for o2 in range(L):
                    best = np.inf
                    for b0 in range(L):
                        for b1 in range(L):
                            for b2 in range(L):
                                val = (
                                    cost[n, b0, b1, b2]
                                    + (o0 - b0 + offs[n, 0]) ** 2
                                    + (o1 - b1 + offs[n, 1]) ** 2
                                    + (o2 - b2 + offs[n, 2]) ** 2
                                )
                                best = min(best, val)
                    assert abs(msg[n, o0, o1, o2] - best) < 1e-3


def test_prims_graph_is_a_valid_tree() -> None:
    rng = np.random.default_rng(0)
    vol = rng.random((32, 32, 32)).astype(np.float32)
    order, parents, edgemst = prims_graph(vol, step=8, grid_shape=(4, 4, 4))
    n = 4 * 4 * 4
    root = int(order[0])
    assert len(set(order.tolist())) == n, "order is not a permutation"
    assert edgemst[root] == 0.0
    for node in range(n):  # every node reaches the root via parents (connected, acyclic)
        seen, c = set(), node
        while c != root:
            assert c not in seen, "cycle in tree"
            seen.add(c)
            c = int(parents[c])


def test_warp_volume_matches_integer_shift() -> None:
    rng = np.random.default_rng(0)
    vol = rng.random((16, 18, 20)).astype(np.float32)
    field = np.zeros((3, 16, 18, 20), np.float32)
    field[0], field[1], field[2] = 2, -3, 1
    from scipy.ndimage import shift as ndshift

    expected = ndshift(vol, (-2, 3, -1), order=1, mode="nearest")
    assert np.abs(warp_volume(vol, field) - expected).max() < 1e-4


def test_compose_consistent_improves_inverse_consistency() -> None:
    rng = np.random.default_rng(0)
    g = (8, 8, 8)
    f = ((rng.random((3,) + g) - 0.5) * 2).astype(np.float32)
    b = -f.copy()
    f2, b2 = compose_consistent(f, b, factor=4)
    before = float(np.abs(_compose(f / 4, b / 4, np) + b / 4).mean())
    after = float(np.abs(_compose(f2 / 4, b2 / 4, np) + b2 / 4).mean())
    assert after < before


def test_upsample_field_preserves_constant() -> None:
    cf = np.zeros((3, 4, 4, 4), np.float32)
    cf[0], cf[1], cf[2] = 5, -2, 3
    uf = upsample_field(cf, (16, 16, 16))
    assert uf.shape == (3, 16, 16, 16)
    assert np.allclose(uf[0], 5) and np.allclose(uf[1], -2) and np.allclose(uf[2], 3)


# --------------------------------------------------------------------------- #
# End-to-end recovery
# --------------------------------------------------------------------------- #

def test_recovers_translation() -> None:
    ref = _texture(0)
    applied = (2, -3, 1)
    mov = apply_integer_shift(ref, applied)
    res = register_deformable(ref, mov, levels_params=_FAST)

    assert isinstance(res, DeformableResult)
    assert res.algorithm_name == ALGORITHM_NAME
    # Field convention: corrected(p)=moving(p+field), so a pure shift gives a
    # near-constant field equal to +applied (the negation of the integer
    # methods' reported correction).
    means = [float(res.field[k].mean()) for k in range(3)]
    assert np.allclose(means, applied, atol=0.6), f"field means {means} vs {applied}"

    corrected = warp_corrected(mov, res)
    drop = 1 - _ssd(ref, corrected) / _ssd(ref, mov)
    assert drop > 0.8, f"SSD only dropped {drop:.0%}"


def test_recovers_smooth_deformation() -> None:
    ref = _texture(1, n=56)
    field_true = _sinusoid_field(56, amp=2.5)
    mov = warp_volume(ref.astype(np.float32), -field_true).astype(np.uint16)

    res = register_deformable(ref, mov, levels_params=_FAST)
    corrected = warp_corrected(mov, res)
    drop = 1 - _ssd(ref, corrected) / _ssd(ref, mov)
    assert drop > 0.7, f"SSD only dropped {drop:.0%}"


def test_gpu_flag_falls_back_to_cpu() -> None:
    ref = _texture(2)
    mov = apply_integer_shift(ref, (1, 2, -2))
    res = register_deformable(ref, mov, levels_params=_FAST, use_gpu=True)
    corrected = warp_corrected(mov, res, use_gpu=True)
    assert _ssd(ref, corrected) < _ssd(ref, mov)


def test_progress_is_monotonic() -> None:
    ref = _texture(3)
    mov = apply_integer_shift(ref, (1, -1, 2))
    fr: list[float] = []
    register_deformable(ref, mov, levels_params=_FAST, progress_callback=fr.append)
    assert fr and fr == sorted(fr) and fr[-1] == 1.0


def test_integer_pipeline_untouched() -> None:
    """The deformable path must not have changed the registry or result type."""
    from shifter.registration import ALGORITHM_REGISTRY
    from shifter.registration.base import RegistrationResult
    import dataclasses

    expected = {
        "Phase Cross-Correlation",
        "Mutual Information",
        "Mutual Information (Brent)",
        "Zero-Normalized Cross-Correlation",
        "deedsBCV (MIND-SSC)",
        "deedsBCV (MIND-SSC, Brent)",
    }
    assert set(ALGORITHM_REGISTRY) == expected, "registry membership changed"
    fields = {f.name for f in dataclasses.fields(RegistrationResult)}
    assert fields == {
        "shift_x", "shift_y", "shift_z",
        "confidence", "raw_metric_value", "algorithm_name",
    }


def test_rejects_small_or_mismatched_volumes() -> None:
    """A too-thin ROI or a shape mismatch raises a clear ValueError up front."""
    small = np.zeros((10, 40, 40), np.uint16)  # 10 < 2*max(grid spacing)=16
    try:
        register_deformable(small, small, levels_params=_FAST)
        raise AssertionError("small volume did not raise")
    except ValueError:
        pass

    a, b = np.zeros((40, 40, 40), np.uint16), np.zeros((40, 40, 32), np.uint16)
    try:
        register_deformable(a, b, levels_params=_FAST)
        raise AssertionError("shape mismatch did not raise")
    except ValueError:
        pass


def test_warp_corrected_rounds_integer_output() -> None:
    """Warping by a zero field returns the input exactly (rounded, not floored)."""
    field = np.zeros((3, 4, 4, 4), np.float32)
    res = DeformableResult(field=field, volume_shape=(16, 16, 16))
    vol = (np.random.default_rng(0).random((16, 16, 16)) * 1000 + 0.6).astype(np.uint16)
    out = warp_corrected(vol, res)
    assert out.dtype == vol.dtype
    assert np.array_equal(out, vol), "identity warp changed integer values"


def test_full_schedule_recovery_slow() -> None:
    """Opt-in faithful full-schedule run (skipped unless SHIFTER_SLOW_TESTS=1)."""
    import os

    if os.environ.get("SHIFTER_SLOW_TESTS") != "1":
        return
    ref = _texture(0, n=64)
    field_true = _sinusoid_field(64, amp=2.5)
    mov = warp_volume(ref.astype(np.float32), -field_true).astype(np.uint16)
    res = register_deformable(ref, mov)
    corrected = warp_corrected(mov, res)
    drop = 1 - _ssd(ref, corrected) / _ssd(ref, mov)
    assert drop > 0.9, f"full-schedule SSD only dropped {drop:.0%}"


_TESTS = [
    test_message_dt_matches_brute_force,
    test_prims_graph_is_a_valid_tree,
    test_warp_volume_matches_integer_shift,
    test_compose_consistent_improves_inverse_consistency,
    test_upsample_field_preserves_constant,
    test_recovers_translation,
    test_recovers_smooth_deformation,
    test_gpu_flag_falls_back_to_cpu,
    test_progress_is_monotonic,
    test_integer_pipeline_untouched,
    test_rejects_small_or_mismatched_volumes,
    test_warp_corrected_rounds_integer_output,
    test_full_schedule_recovery_slow,
]


def run_validation() -> bool:
    print("=" * 60)
    print("deedsBCV (deformable) Validation")
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
