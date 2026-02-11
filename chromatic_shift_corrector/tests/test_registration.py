"""Validation script for auto-registration algorithms.

Generates synthetic 3-channel test volumes with **known shifts**, runs each
registration algorithm, and verifies that the detected shifts match the
ground truth exactly (integer-shift accuracy).

Run via::

    python -m chromatic_shift_corrector.tests.test_registration
"""

from __future__ import annotations

import sys
import time

import numpy as np

from chromatic_shift_corrector.registration import (
    ALGORITHM_REGISTRY,
    RegistrationResult,
    preprocess,
)

# ---------------------------------------------------------------------------
# Ground-truth shifts (what was applied to create the moved channels).
# The registration should detect the *negated* shifts (the correction).
# ---------------------------------------------------------------------------
GROUND_TRUTH_SHIFTS = {
    1: (5, -3, 12),    # ch1: applied Z=+5, Y=-3, X=+12  → correction Z=-5, Y=+3, X=-12
    2: (-2, 7, -8),    # ch2: applied Z=-2, Y=+7, X=-8   → correction Z=+2, Y=-7, X=+8
}

# Volume parameters.
VOLUME_SHAPE = (128, 128, 128)  # (Z, Y, X)
NUM_BLOBS = 15
BLOB_RADIUS_RANGE = (6, 14)
NOISE_LEVEL = 200  # Poisson noise scaling


def _make_blob_field(
    shape: tuple[int, int, int],
    n_blobs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a 3-D field of Gaussian blobs (shared structure)."""
    vol = np.zeros(shape, dtype=np.float64)
    for _ in range(n_blobs):
        cz = rng.integers(20, shape[0] - 20)
        cy = rng.integers(20, shape[1] - 20)
        cx = rng.integers(20, shape[2] - 20)
        sigma = rng.uniform(BLOB_RADIUS_RANGE[0], BLOB_RADIUS_RANGE[1])
        intensity = rng.uniform(5000, 40000)

        zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
        dist2 = (
            (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
        )
        blob = intensity * np.exp(-dist2 / (2.0 * sigma ** 2))
        vol += blob
    return vol


def _apply_shift(vol: np.ndarray, shift_zyx: tuple[int, int, int]) -> np.ndarray:
    """Apply integer shift with zero-padding."""
    sz, sy, sx = shift_zyx
    result = np.zeros_like(vol)
    nz, ny, nx = vol.shape

    def _slices(shift: int, length: int):
        if shift > 0:
            return slice(0, max(length - shift, 0)), slice(shift, length)
        elif shift < 0:
            return slice(-shift, length), slice(0, max(length + shift, 0))
        return slice(0, length), slice(0, length)

    sz_src, sz_dst = _slices(sz, nz)
    sy_src, sy_dst = _slices(sy, ny)
    sx_src, sx_dst = _slices(sx, nx)
    result[sz_dst, sy_dst, sx_dst] = vol[sz_src, sy_src, sx_src]
    return result


def _add_poisson_noise(vol: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add Poisson noise to simulate microscopy data."""
    # Scale so noise level is realistic.
    scaled = np.clip(vol, 0, None)
    # Poisson parameter = signal intensity.
    noisy = rng.poisson(lam=np.clip(scaled / NOISE_LEVEL, 0.1, None)).astype(np.float64)
    noisy *= NOISE_LEVEL
    return np.clip(noisy, 0, 65535).astype(np.uint16)


def generate_test_data() -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Create reference and shifted channel volumes.

    Returns
    -------
    reference : np.ndarray
        Reference channel volume (uint16).
    channels : dict[int, np.ndarray]
        Mapping of channel index → shifted volume (uint16).
    """
    rng = np.random.default_rng(42)

    # Shared structure (autofluorescence-like).
    shared = _make_blob_field(VOLUME_SHAPE, NUM_BLOBS, rng)

    # Channel-specific features (different intensities per channel).
    ch_specific = {}
    for ch_i in [0, 1, 2]:
        extra = _make_blob_field(VOLUME_SHAPE, 5, rng) * rng.uniform(0.3, 0.7)
        ch_specific[ch_i] = extra

    # Reference channel (ch0).
    ref_raw = shared + ch_specific[0]
    reference = _add_poisson_noise(ref_raw, rng)

    # Shifted channels.
    channels = {}
    for ch_i, (dz, dy, dx) in GROUND_TRUTH_SHIFTS.items():
        raw = shared + ch_specific[ch_i]
        raw_uint16 = _add_poisson_noise(raw, rng)
        channels[ch_i] = _apply_shift(raw_uint16, (dz, dy, dx))

    return reference, channels


def run_validation() -> bool:
    """Run all three algorithms on synthetic data and report results.

    Returns True if all algorithms pass.
    """
    print("=" * 60)
    print("Auto-Registration Validation")
    print("=" * 60)
    print(f"Volume shape: {VOLUME_SHAPE}")
    print(f"Ground-truth shifts:")
    for ch_i, (dz, dy, dx) in GROUND_TRUTH_SHIFTS.items():
        print(f"  ch{ch_i}: Z={dz:+d}, Y={dy:+d}, X={dx:+d}")
        print(f"    Expected correction: Z={-dz:+d}, Y={-dy:+d}, X={-dx:+d}")
    print()

    print("Generating synthetic test data...")
    t0 = time.time()
    reference, channels = generate_test_data()
    print(f"  Done in {time.time() - t0:.1f}s")
    print()

    search_range_xy = 20
    search_range_z = 50
    all_passed = True

    for algo_name, algo_cls in ALGORITHM_REGISTRY.items():
        print("-" * 60)
        print(f"Algorithm: {algo_name}")
        print("-" * 60)

        algo = algo_cls()
        algo_passed = True

        for ch_i, shifted_vol in channels.items():
            dz, dy, dx = GROUND_TRUTH_SHIFTS[ch_i]
            expected_x = -dx
            expected_y = -dy
            expected_z = -dz

            t0 = time.time()
            result: RegistrationResult = algo.register(
                reference, shifted_vol, search_range_xy, search_range_z, use_gpu=False
            )
            elapsed = time.time() - t0

            match_x = result.shift_x == expected_x
            match_y = result.shift_y == expected_y
            match_z = result.shift_z == expected_z
            passed = match_x and match_y and match_z

            status = "PASS" if passed else "FAIL"
            print(f"  ch{ch_i}: {status}")
            print(f"    Expected:  X={expected_x:+d}, Y={expected_y:+d}, Z={expected_z:+d}")
            print(f"    Detected:  X={result.shift_x:+d}, Y={result.shift_y:+d}, Z={result.shift_z:+d}")
            print(f"    Confidence: {result.confidence:.3f}")
            print(f"    Raw metric: {result.raw_metric_value:.4f}")
            print(f"    Time: {elapsed:.2f}s")

            if not passed:
                diff_x = result.shift_x - expected_x
                diff_y = result.shift_y - expected_y
                diff_z = result.shift_z - expected_z
                print(f"    Error: dX={diff_x:+d}, dY={diff_y:+d}, dZ={diff_z:+d}")
                algo_passed = False

        if algo_passed:
            print(f"  >> {algo_name}: ALL CHANNELS PASSED")
        else:
            print(f"  >> {algo_name}: SOME CHANNELS FAILED")
            all_passed = False
        print()

    print("=" * 60)
    if all_passed:
        print("OVERALL: ALL ALGORITHMS PASSED")
    else:
        print("OVERALL: SOME ALGORITHMS FAILED")
    print("=" * 60)

    return all_passed


def main() -> None:
    passed = run_validation()
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
