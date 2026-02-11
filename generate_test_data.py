#!/usr/bin/env python
"""Generate synthetic test datasets for validating the chromatic shift corrector.

Creates 2-3 channel 16-bit BigTIFF volumes with a known geometric pattern
(bright sphere + grid) and applies known chromatic shifts so the correction
tool can be validated by reversing them.

Usage
-----
    python generate_test_data.py [--output-dir DIR] [--size 256] [--n-channels 3]

The script writes:
    ch0.tif  — reference channel (unshifted)
    ch1.tif  — shifted by known amounts
    ch2.tif  — shifted by different known amounts (if 3 channels)
    ground_truth_shifts.json  — the shifts applied, for verification
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile


def make_sphere(shape: tuple[int, int, int], center: tuple[int, int, int], radius: int) -> np.ndarray:
    """Create a binary sphere mask."""
    zz, yy, xx = np.ogrid[
        : shape[0], : shape[1], : shape[2]
    ]
    dist = np.sqrt(
        (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2
    )
    return (dist <= radius).astype(np.float64)


def make_grid(shape: tuple[int, int, int], spacing: int = 32) -> np.ndarray:
    """Create a 3D grid pattern."""
    vol = np.zeros(shape, dtype=np.float64)
    vol[::spacing, :, :] = 1.0
    vol[:, ::spacing, :] = 1.0
    vol[:, :, ::spacing] = 1.0
    return vol


def make_pattern(shape: tuple[int, int, int]) -> np.ndarray:
    """Create a combined sphere + grid test pattern as uint16."""
    cz, cy, cx = shape[0] // 2, shape[1] // 2, shape[2] // 2
    radius = min(shape) // 4

    sphere = make_sphere(shape, (cz, cy, cx), radius)
    grid = make_grid(shape, spacing=max(32, min(shape) // 8))

    # Combine: sphere is bright, grid is moderate intensity.
    combined = sphere * 50000 + grid * 20000
    return np.clip(combined, 0, 65535).astype(np.uint16)


def apply_shift(vol: np.ndarray, shift_zyx: tuple[int, int, int]) -> np.ndarray:
    """Apply a shift with zero-padding (no wrap)."""
    sz, sy, sx = shift_zyx
    result = np.zeros_like(vol)
    nz, ny, nx = vol.shape

    def _slices(shift: int, length: int) -> tuple[slice, slice]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic test data")
    parser.add_argument(
        "--output-dir", type=str, default="test_data",
        help="Output directory (default: test_data)",
    )
    parser.add_argument(
        "--size", type=int, default=256,
        help="Volume size in each dimension (default: 256)",
    )
    parser.add_argument(
        "--nz", type=int, default=None,
        help="Z dimension (default: same as --size)",
    )
    parser.add_argument(
        "--n-channels", type=int, default=3, choices=[2, 3],
        help="Number of channels (default: 3)",
    )
    args = parser.parse_args()

    nz = args.nz or args.size
    shape = (nz, args.size, args.size)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.n_channels}-channel test data, shape={shape}")

    pattern = make_pattern(shape)

    # Define known shifts (what the correction tool should reverse).
    shifts = [
        (0, 0, 0),       # ch0: reference (no shift)
        (5, 3, -2),      # ch1: shifted by Z=+5, Y=+3, X=-2
    ]
    if args.n_channels == 3:
        shifts.append((-3, -1, 4))  # ch2: shifted by Z=-3, Y=-1, X=+4

    metadata = {
        "volume_shape_zyx": list(shape),
        "channels": [],
    }

    for i, shift in enumerate(shifts):
        fname = f"ch{i}.tif"
        shifted = apply_shift(pattern, shift) if any(s != 0 for s in shift) else pattern.copy()
        path = outdir / fname
        tifffile.imwrite(str(path), shifted, bigtiff=True, photometric="minisblack")
        print(f"  Wrote {path} (shift Z={shift[0]:+d}, Y={shift[1]:+d}, X={shift[2]:+d})")

        metadata["channels"].append({
            "filename": fname,
            "channel_index": i,
            "applied_shift_z": shift[0],
            "applied_shift_y": shift[1],
            "applied_shift_x": shift[2],
            "correction_shift_z": -shift[0],
            "correction_shift_y": -shift[1],
            "correction_shift_x": -shift[2],
        })

    meta_path = outdir / "ground_truth_shifts.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Wrote {meta_path}")
    print(
        "\nTo validate: load these files in the corrector, set shifts to the "
        "'correction_shift_*' values from ground_truth_shifts.json, and verify "
        "that the channels align."
    )


if __name__ == "__main__":
    main()
