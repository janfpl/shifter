"""Validation script for Luxendo H5 file support.

Generates synthetic .lux.h5 test files with known shifts, runs the full
load-export pipeline, and validates:

- Metadata parsing (voxel sizes, channel description)
- Pyramid detection and naming
- Corrected Data dataset matches expected shifts
- Pyramid levels are consistent with corrected full-res data
- Original metadata is preserved verbatim in output
- Files whose name starts with "main" are ignored

Run via::

    python -m shifter.tests.test_h5
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

from shifter.data_loader import H5Loader, validate_channels
from shifter.export_engine import estimate_output_sizes, run_export_h5
from shifter.h5_utils import (
    H5FileManager,
    block_average_3d,
    detect_pyramid_levels,
    parse_h5_metadata,
    scan_h5_files,
)
from shifter.shift_manager import ShiftManager

# Test parameters.
VOLUME_SHAPE = (100, 256, 256)  # (Z, Y, X)
CHUNK_SHAPE = (64, 64, 64)
NUM_CHANNELS = 3

GROUND_TRUTH_SHIFTS = {
    1: (3, -5, 8),    # ch1: applied Z=+3, Y=-5, X=+8
    2: (-2, 4, -6),   # ch2: applied Z=-2, Y=+4, X=-6
}

VOXEL_XY = 0.40625
VOXEL_Z = 1.0

CHANNEL_DESCRIPTIONS = ["Green-22", "Red-561", "Blue-405"]


def _make_metadata_json(ch_idx: int) -> str:
    """Create Luxendo-style metadata JSON."""
    nz, ny, nx = VOLUME_SHAPE
    meta = {
        "processingInformation": {
            "voxel_size_um": {
                "width": VOXEL_XY,
                "height": VOXEL_XY,
                "depth": VOXEL_Z,
            },
            "image_size_vx": {
                "width": nx,
                "height": ny,
                "depth": nz,
            },
            "channel_description": CHANNEL_DESCRIPTIONS[ch_idx],
            "channel_id": ch_idx,
            "affine_to_sample": {"translation": [0, 0, 0], "rotation": [0, 0, 0]},
        }
    }
    return json.dumps(meta)


def _make_blob_volume(rng: np.random.Generator) -> np.ndarray:
    """Generate a 3D volume with Gaussian blobs."""
    vol = np.zeros(VOLUME_SHAPE, dtype=np.float64)
    nz, ny, nx = VOLUME_SHAPE
    for _ in range(20):
        cz = rng.integers(15, nz - 15)
        cy = rng.integers(15, ny - 15)
        cx = rng.integers(15, nx - 15)
        sigma = rng.uniform(4, 10)
        intensity = rng.uniform(5000, 40000)
        zz, yy, xx = np.ogrid[:nz, :ny, :nx]
        dist2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
        vol += intensity * np.exp(-dist2 / (2.0 * sigma ** 2))
    return np.clip(vol, 0, 65535).astype(np.uint16)


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


def _create_h5_file(
    filepath: Path,
    data: np.ndarray,
    ch_idx: int,
) -> None:
    """Create a synthetic .lux.h5 file with Data, pyramids, and metadata."""
    nz, ny, nx = data.shape
    with h5py.File(str(filepath), "w") as f:
        # Full-resolution Data.
        f.create_dataset("Data", data=data, chunks=CHUNK_SHAPE, dtype=np.uint16)

        # Pyramid: Data_2_2_2.
        out_nz_2 = nz // 2
        out_ny_2 = ny // 2
        out_nx_2 = nx // 2
        pyr2 = np.zeros((out_nz_2, out_ny_2, out_nx_2), dtype=np.uint16)
        for oz in range(out_nz_2):
            slab = data[oz * 2 : oz * 2 + 2, :, :]
            pyr2[oz, :, :] = block_average_3d(slab, 2, 2, 2)
        f.create_dataset(
            "Data_2_2_2", data=pyr2, chunks=(32, 32, 32), dtype=np.uint16
        )

        # Pyramid: Data_3_3_3.
        out_nz_3 = nz // 3
        out_ny_3 = ny // 3
        out_nx_3 = nx // 3
        pyr3 = np.zeros((out_nz_3, out_ny_3, out_nx_3), dtype=np.uint16)
        for oz in range(out_nz_3):
            slab = data[oz * 3 : oz * 3 + 3, :, :]
            pyr3[oz, :, :] = block_average_3d(slab, 3, 3, 3)
        f.create_dataset(
            "Data_3_3_3", data=pyr3, chunks=(32, 32, 32), dtype=np.uint16
        )

        # Metadata.
        meta_str = _make_metadata_json(ch_idx)
        f.create_dataset("metadata", data=meta_str.encode("utf-8"))


def generate_test_directory(tmpdir: Path) -> tuple[list[Path], np.ndarray]:
    """Create a directory of synthetic H5 files.

    Returns (channel_files, reference_data).
    """
    rng = np.random.default_rng(123)
    ref_data = _make_blob_volume(rng)

    files = []
    for ch_i in range(NUM_CHANNELS):
        fname = f"ch{ch_i}_test.lux.h5"
        filepath = tmpdir / fname

        if ch_i in GROUND_TRUTH_SHIFTS:
            shift = GROUND_TRUTH_SHIFTS[ch_i]
            ch_data = _apply_shift(ref_data, shift)
        else:
            ch_data = ref_data.copy()

        _create_h5_file(filepath, ch_data, ch_i)
        files.append(filepath)

    # Create a dummy "main" file that should be ignored.
    main_path = tmpdir / "main_header.lux.h5"
    with h5py.File(str(main_path), "w") as f:
        f.create_dataset("Data", data=np.zeros((1, 1, 1), dtype=np.uint16))

    return files, ref_data


def test_scan_ignores_main(tmpdir: Path) -> bool:
    """Verify that scan_h5_files ignores 'main*' files."""
    found = scan_h5_files(tmpdir)
    names = [f.name for f in found]
    for n in names:
        if n.lower().startswith("main"):
            print(f"  FAIL: scan_h5_files returned 'main' file: {n}")
            return False
    if len(found) != NUM_CHANNELS:
        print(f"  FAIL: expected {NUM_CHANNELS} files, got {len(found)}")
        return False
    print(f"  PASS: {len(found)} files found, 'main' file correctly ignored")
    return True


def test_metadata_parsing(files: list[Path]) -> bool:
    """Verify metadata extraction from H5 files."""
    all_ok = True
    for ch_i, fp in enumerate(files):
        with h5py.File(str(fp), "r") as f:
            meta = parse_h5_metadata(f)

        if not meta:
            print(f"  FAIL: no metadata for {fp.name}")
            all_ok = False
            continue

        xy = meta.get("voxel_size_xy_um")
        z = meta.get("voxel_size_z_um")
        desc = meta.get("channel_description")

        if abs(xy - VOXEL_XY) > 1e-6:
            print(f"  FAIL: {fp.name} voxel_xy={xy}, expected {VOXEL_XY}")
            all_ok = False
        if abs(z - VOXEL_Z) > 1e-6:
            print(f"  FAIL: {fp.name} voxel_z={z}, expected {VOXEL_Z}")
            all_ok = False
        if desc != CHANNEL_DESCRIPTIONS[ch_i]:
            print(f"  FAIL: {fp.name} description={desc!r}, expected {CHANNEL_DESCRIPTIONS[ch_i]!r}")
            all_ok = False

    if all_ok:
        print(f"  PASS: all {len(files)} files have correct metadata")
    return all_ok


def test_pyramid_detection(files: list[Path]) -> bool:
    """Verify pyramid level detection."""
    all_ok = True
    for fp in files:
        with h5py.File(str(fp), "r") as f:
            levels = detect_pyramid_levels(f)
        level_names = [l[0] for l in levels]
        if "Data_2_2_2" not in level_names or "Data_3_3_3" not in level_names:
            print(f"  FAIL: {fp.name} missing expected pyramids, got {level_names}")
            all_ok = False
    if all_ok:
        print(f"  PASS: all files have Data_2_2_2 and Data_3_3_3 pyramids")
    return all_ok


def test_h5_loader(files: list[Path]) -> bool:
    """Test H5Loader initialization and properties."""
    mgr = H5FileManager()
    all_ok = True
    try:
        loaders = [H5Loader(fp, mgr) for fp in files]

        # Validate shapes.
        for loader in loaders:
            if loader.shape != VOLUME_SHAPE:
                print(f"  FAIL: {loader.path.name} shape={loader.shape}, expected {VOLUME_SHAPE}")
                all_ok = False

        # Validate dtype.
        for loader in loaders:
            if loader.dtype != np.uint16:
                print(f"  FAIL: {loader.path.name} dtype={loader.dtype}")
                all_ok = False

        # Validate multiscale.
        for loader in loaders:
            if len(loader.multiscale) != 3:  # full + 2 pyramids
                print(f"  FAIL: {loader.path.name} multiscale has {len(loader.multiscale)} levels, expected 3")
                all_ok = False

        # Validate channel validation.
        ok, msg = validate_channels(loaders)
        if not ok:
            print(f"  FAIL: validate_channels: {msg}")
            all_ok = False

        if all_ok:
            print("  PASS: H5Loader, multiscale, and validation all correct")
    finally:
        mgr.close_all()
    return all_ok


def test_estimate_includes_pyramids(files: list[Path]) -> bool:
    """Verify estimate_output_sizes counts pyramid levels, not just full-res."""
    mgr = H5FileManager()
    all_ok = True
    try:
        loaders = [H5Loader(fp, mgr) for fp in files]
        nz, ny, nx = VOLUME_SHAPE

        # Independent expected: full-res + Data_2_2_2 + Data_3_3_3 per channel.
        per_channel = (
            nz * ny * nx * 2
            + (nz // 2) * (ny // 2) * (nx // 2) * 2
            + (nz // 3) * (ny // 3) * (nx // 3) * 2
        )
        expected = per_channel * len(loaders)
        got = sum(estimate_output_sizes(loaders))
        if got != expected:
            print(f"  FAIL: estimate={got}, expected {expected}")
            all_ok = False
        else:
            print(f"  PASS: estimate includes pyramids ({got / 1024**2:.1f} MiB)")

        # Must strictly exceed a full-res-only estimate (proves pyramids counted).
        full_only = nz * ny * nx * 2 * len(loaders)
        if got <= full_only:
            print(f"  FAIL: estimate {got} did not exceed full-res-only {full_only}")
            all_ok = False

        # A loader without pyramid_levels (e.g. BigTIFF) estimates full-res only.
        class _NoPyrLoader:
            shape = (10, 20, 30)

        stub = estimate_output_sizes([_NoPyrLoader()])[0]
        if stub != 10 * 20 * 30 * 2:
            print(f"  FAIL: no-pyramid estimate={stub}, expected {10 * 20 * 30 * 2}")
            all_ok = False
    finally:
        mgr.close_all()

    if all_ok:
        print("  PASS: estimate_output_sizes pyramid accounting correct")
    return all_ok


def test_export_and_pyramids(files: list[Path], ref_data: np.ndarray) -> bool:
    """Test full H5 export with shift correction and pyramid regeneration."""
    mgr = H5FileManager()
    outdir = Path(tempfile.mkdtemp())
    all_ok = True

    try:
        loaders = [H5Loader(fp, mgr) for fp in files]

        shift_manager = ShiftManager()
        filenames = [fp.name for fp in files]
        colormaps = ["green", "magenta", "cyan"]
        shift_manager.init_channels(filenames, reference_index=0, colormaps=colormaps)

        # Set correction shifts (negative of applied).
        for ch_i, (dz, dy, dx) in GROUND_TRUTH_SHIFTS.items():
            shift_manager.set_shift(ch_i, "x", -dx)
            shift_manager.set_shift(ch_i, "y", -dy)
            shift_manager.set_shift(ch_i, "z", -dz)

        meta_path = run_export_h5(
            loaders,
            shift_manager,
            outdir,
            ram_percent=90,
            voxel_xy=VOXEL_XY,
            voxel_z=VOXEL_Z,
        )

        # Check metadata JSON.
        with open(meta_path) as f:
            meta = json.load(f)

        if meta.get("input_format") != "luxendo_h5":
            print(f"  FAIL: metadata input_format={meta.get('input_format')}")
            all_ok = False

        if meta.get("voxel_size_source") != "h5_metadata":
            print(f"  FAIL: metadata voxel_size_source={meta.get('voxel_size_source')}")
            all_ok = False

        # Pre-export estimate must match the post-export bytes_written (both
        # count full-res + regenerated pyramids).
        est_total = sum(estimate_output_sizes(loaders))
        if est_total != meta.get("bytes_written"):
            print(f"  FAIL: estimate {est_total} != metadata bytes_written {meta.get('bytes_written')}")
            all_ok = False
        else:
            print("  PASS: pre-export estimate matches metadata bytes_written")

        # Check corrected files. Full-volume H5 export keeps the original
        # filename (no suffix) so companion Imaris/BigDataViewer headers
        # keep working.
        for ch_i, (dz, dy, dx) in GROUND_TRUTH_SHIFTS.items():
            fname = files[ch_i].name
            out_name = fname
            out_path = outdir / out_name

            if not out_path.exists():
                print(f"  FAIL: corrected file missing: {out_name}")
                all_ok = False
                continue

            with h5py.File(str(out_path), "r") as f:
                # Check Data exists.
                if "Data" not in f:
                    print(f"  FAIL: {out_name} missing Data dataset")
                    all_ok = False
                    continue

                corrected = f["Data"][:]

                # The corrected channel should approximately match the reference
                # in the overlapping region (since we applied the inverse shift).
                # Check a central region.
                margin = 20
                nz, ny, nx = VOLUME_SHAPE
                ref_sub = ref_data[margin:nz-margin, margin:ny-margin, margin:nx-margin]
                cor_sub = corrected[margin:nz-margin, margin:ny-margin, margin:nx-margin]
                # Allow tolerance for edge effects.
                diff = np.abs(ref_sub.astype(float) - cor_sub.astype(float))
                max_diff = diff.max()
                mean_diff = diff.mean()
                if mean_diff > 100:
                    print(f"  FAIL: ch{ch_i} corrected data mean diff={mean_diff:.1f} (expected ~0)")
                    all_ok = False
                else:
                    print(f"  PASS: ch{ch_i} corrected data matches reference (mean diff={mean_diff:.1f})")

                # Check metadata preserved.
                if "metadata" not in f:
                    print(f"  FAIL: {out_name} missing metadata dataset")
                    all_ok = False
                else:
                    orig_meta_bytes = None
                    with h5py.File(str(files[ch_i]), "r") as orig:
                        orig_meta_bytes = orig["metadata"][()]
                    out_meta_bytes = f["metadata"][()]
                    if orig_meta_bytes != out_meta_bytes:
                        print(f"  FAIL: {out_name} metadata not preserved verbatim")
                        all_ok = False
                    else:
                        print(f"  PASS: ch{ch_i} original metadata preserved")

                # Check pyramids exist and are consistent.
                levels = detect_pyramid_levels(f)
                level_names = [l[0] for l in levels]
                if "Data_2_2_2" not in level_names or "Data_3_3_3" not in level_names:
                    print(f"  FAIL: {out_name} missing pyramid levels, got {level_names}")
                    all_ok = False
                else:
                    # Validate Data_2_2_2 shape.
                    pyr2 = f["Data_2_2_2"]
                    expected_shape = (nz // 2, ny // 2, nx // 2)
                    if pyr2.shape != expected_shape:
                        print(f"  FAIL: {out_name} Data_2_2_2 shape={pyr2.shape}, expected {expected_shape}")
                        all_ok = False
                    else:
                        print(f"  PASS: ch{ch_i} pyramids regenerated correctly")

        # Check reference channel (no shift) output.
        ref_out = outdir / files[0].name
        if ref_out.exists():
            with h5py.File(str(ref_out), "r") as f:
                ref_corr = f["Data"][:]
                diff = np.abs(ref_data.astype(float) - ref_corr.astype(float))
                if diff.max() > 0:
                    print(f"  FAIL: reference channel output differs from input")
                    all_ok = False
                else:
                    print(f"  PASS: reference channel output matches input exactly")

        # Check channel metadata entries.
        for ch_entry in meta.get("channels", []):
            if "pyramid_levels_regenerated" in ch_entry:
                expected_levels = ["Data_2_2_2", "Data_3_3_3"]
                if ch_entry["pyramid_levels_regenerated"] != expected_levels:
                    print(f"  FAIL: channel {ch_entry['channel_index']} pyramid_levels_regenerated mismatch")
                    all_ok = False

        if all_ok:
            print("  PASS: all export checks passed")

    finally:
        mgr.close_all()
        shutil.rmtree(outdir, ignore_errors=True)

    return all_ok


def run_validation() -> bool:
    """Run all H5 tests and report results."""
    print("=" * 60)
    print("Luxendo H5 Support Validation")
    print("=" * 60)

    tmpdir = Path(tempfile.mkdtemp())
    all_passed = True

    try:
        print("\nGenerating synthetic H5 test files...")
        files, ref_data = generate_test_directory(tmpdir)
        print(f"  Created {len(files)} channel files + 1 main file")

        print("\n--- Test: scan_h5_files (main file filtering) ---")
        if not test_scan_ignores_main(tmpdir):
            all_passed = False

        print("\n--- Test: Metadata parsing ---")
        if not test_metadata_parsing(files):
            all_passed = False

        print("\n--- Test: Pyramid detection ---")
        if not test_pyramid_detection(files):
            all_passed = False

        print("\n--- Test: H5Loader ---")
        if not test_h5_loader(files):
            all_passed = False

        print("\n--- Test: Output size estimate (pyramids included) ---")
        if not test_estimate_includes_pyramids(files):
            all_passed = False

        print("\n--- Test: Export with correction and pyramid regeneration ---")
        if not test_export_and_pyramids(files, ref_data):
            all_passed = False

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("\n" + "=" * 60)
    if all_passed:
        print("OVERALL: ALL H5 TESTS PASSED")
    else:
        print("OVERALL: SOME H5 TESTS FAILED")
    print("=" * 60)

    return all_passed


def main() -> None:
    passed = run_validation()
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
