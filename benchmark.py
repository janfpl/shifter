#!/usr/bin/env python
"""Benchmark script for measuring CPU-parallelisation performance.

Generates synthetic 3-channel test volumes, runs registration with all
three algorithms, exports corrected BigTIFF files, and writes a
``performance_log.txt`` with timestamped start/end markers for every
phase.

Usage
-----
    python benchmark.py [--size 128] [--search-xy 10] [--search-z 5]

The script produces:
    benchmark_output/
        performance_log.txt   ← main deliverable
        ch*_corrected.tif     ← exported BigTIFF files
        correction_metadata.json
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import tifffile

# ---------------------------------------------------------------------------
# Helpers (copied from generate_test_data.py to keep the script self-contained)
# ---------------------------------------------------------------------------

def _make_sphere(shape, center, radius):
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist = np.sqrt(
        (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2
    )
    return (dist <= radius).astype(np.float64)


def _make_grid(shape, spacing=32):
    vol = np.zeros(shape, dtype=np.float64)
    vol[::spacing, :, :] = 1.0
    vol[:, ::spacing, :] = 1.0
    vol[:, :, ::spacing] = 1.0
    return vol


def _make_pattern(shape):
    cz, cy, cx = shape[0] // 2, shape[1] // 2, shape[2] // 2
    radius = min(shape) // 4
    sphere = _make_sphere(shape, (cz, cy, cx), radius)
    grid = _make_grid(shape, spacing=max(32, min(shape) // 8))
    combined = sphere * 50000 + grid * 20000
    return np.clip(combined, 0, 65535).astype(np.uint16)


def _apply_shift(vol, shift_zyx):
    sz, sy, sx = shift_zyx
    result = np.zeros_like(vol)
    nz, ny, nx = vol.shape

    def _slices(shift, length):
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


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CPU parallelisation")
    parser.add_argument("--size", type=int, default=128, help="XY dimension (default: 128)")
    parser.add_argument("--nz", type=int, default=None, help="Z dimension (default: same as --size)")
    parser.add_argument("--search-xy", type=int, default=10, help="XY search range (default: 10)")
    parser.add_argument("--search-z", type=int, default=5, help="Z search range (default: 5)")
    parser.add_argument("--output-dir", type=str, default="benchmark_output",
                        help="Output directory (default: benchmark_output)")
    args = parser.parse_args()

    nz = args.nz or args.size
    shape = (nz, args.size, args.size)
    outdir = Path(args.output_dir)

    # Clean previous run.
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)

    # ------------------------------------------------------------------
    # 1.  Generate test data
    # ------------------------------------------------------------------
    print(f"=== Generating 3-channel test data, shape={shape} ===")
    pattern = _make_pattern(shape)
    shifts = [(0, 0, 0), (3, 2, -1), (-2, -1, 3)]

    data_dir = outdir / "data"
    data_dir.mkdir()
    for i, shift in enumerate(shifts):
        vol = _apply_shift(pattern, shift) if any(s != 0 for s in shift) else pattern.copy()
        path = data_dir / f"ch{i}.tif"
        tifffile.imwrite(str(path), vol, bigtiff=True, photometric="minisblack")
        print(f"  Wrote {path}  shift=({shift[0]:+d},{shift[1]:+d},{shift[2]:+d})")

    # ------------------------------------------------------------------
    # 2.  Set up performance logging
    # ------------------------------------------------------------------
    from chromatic_shift_corrector.perf_logger import setup_perf_log, timed_operation, log_event

    log_path = setup_perf_log(outdir)
    log_event(f"Benchmark started | volume shape={shape} "
              f"search_xy={args.search_xy} search_z={args.search_z} "
              f"CPU cores={os.cpu_count()}")
    print(f"  Performance log: {log_path}")

    # ------------------------------------------------------------------
    # 3.  Load volumes as numpy arrays (simulating what the app does)
    # ------------------------------------------------------------------
    volumes = []
    for i in range(3):
        path = data_dir / f"ch{i}.tif"
        vol = tifffile.imread(str(path))
        volumes.append(vol)
    ref_vol = volumes[0].astype(np.float64)
    mov_vol = volumes[1].astype(np.float64)

    # ------------------------------------------------------------------
    # 4.  Benchmark each registration algorithm
    # ------------------------------------------------------------------
    from chromatic_shift_corrector.registration import ALGORITHM_REGISTRY

    print("\n=== Registration benchmarks ===")

    for algo_name, algo_cls in ALGORITHM_REGISTRY.items():
        algo = algo_cls() if algo_name != "Phase Cross-Correlation" else algo_cls(normalization="phase")
        print(f"\n  [{algo_name}]")

        with timed_operation(f"Registration: {algo_name}"):
            t0 = time.perf_counter()
            result = algo.register(
                ref_vol, mov_vol,
                args.search_xy, args.search_z,
                use_gpu=False,
            )
            elapsed = time.perf_counter() - t0

        print(f"    shift = ({result.shift_z:+d}, {result.shift_y:+d}, {result.shift_x:+d})")
        print(f"    confidence = {result.confidence:.4f}")
        print(f"    elapsed = {elapsed:.3f}s")
        log_event(f"Result {algo_name}: shift=({result.shift_z},{result.shift_y},{result.shift_x}) "
                  f"confidence={result.confidence:.4f} elapsed={elapsed:.3f}s")

    # ------------------------------------------------------------------
    # 5.  Benchmark BigTIFF export
    # ------------------------------------------------------------------
    print("\n=== Export benchmark (BigTIFF) ===")

    import dask.array as da
    from chromatic_shift_corrector.shift_manager import ShiftManager, ChannelTransform
    from chromatic_shift_corrector.export_engine import export_channel, compute_chunk_size

    export_dir = outdir / "exported"
    export_dir.mkdir()

    # Create a simple ShiftManager-like setup for the export.
    known_corrections = [(0, 0, 0), (-3, -2, 1), (2, 1, -3)]

    for i in range(3):
        dask_vol = da.from_array(volumes[i], chunks=(64, -1, -1))
        sz, sy, sx = known_corrections[i]
        transform = ChannelTransform(
            filename=f"ch{i}.tif",
            channel_index=i,
            shift_x=sx,
            shift_y=sy,
            shift_z=sz,
        )

        out_path = export_dir / f"ch{i}_corrected.tif"
        chunk_z = compute_chunk_size((shape[1], shape[2]), 3, ram_percent=90)

        with timed_operation(f"Export TIFF: ch{i}.tif"):
            t0 = time.perf_counter()
            export_channel(dask_vol, transform, out_path, chunk_z)
            elapsed = time.perf_counter() - t0

        file_size = out_path.stat().st_size / (1024 * 1024)
        print(f"  ch{i}_corrected.tif: {file_size:.1f} MB in {elapsed:.3f}s")

    # ------------------------------------------------------------------
    # 6.  Summary
    # ------------------------------------------------------------------
    print(f"\n=== Done ===")
    print(f"Performance log: {log_path}")
    print(f"Output dir:      {outdir}")

    # Print the log contents.
    print(f"\n{'='*72}")
    print("PERFORMANCE LOG CONTENTS:")
    print(f"{'='*72}")
    with open(log_path) as f:
        print(f.read())


if __name__ == "__main__":
    main()
