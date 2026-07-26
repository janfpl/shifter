"""Validation for streaming (fused) pyramid generation.

The streaming writer builds every pyramid level from the corrected export slabs
while they are in memory, instead of rereading the written ``Data`` dataset once
per level. These tests pin the property that matters: its output must be
**bit-identical** to the original reread-based :func:`generate_pyramid_level`,
across hierarchical and non-hierarchical factor ladders, odd shapes that force
trimming, and slab depths that do not divide the depth factors.

Runnable via pytest, or standalone::

    python -m shifter.tests.test_pyramids
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

from shifter.h5_utils import (
    StreamingPyramidWriter,
    block_average_3d,
    block_sum_3d,
    generate_pyramid_level,
    pyramid_sum_dtype,
)


def _reference(volume: np.ndarray, levels, path: Path) -> dict[str, np.ndarray]:
    """Build pyramids the original way (write Data, then reread per level)."""
    with h5py.File(str(path), "w") as f:
        f.create_dataset("Data", data=volume, chunks=None)
        for name, fw, fh, fd in levels:
            generate_pyramid_level(f, name, fw, fh, fd)
        return {name: f[name][()] for name, *_ in levels if name in f}


def _streamed(volume: np.ndarray, levels, path: Path, slab_z: int) -> dict[str, np.ndarray]:
    """Build pyramids the new way (feed slabs as they are written)."""
    nz = volume.shape[0]
    with h5py.File(str(path), "w") as f:
        ds = f.create_dataset("Data", shape=volume.shape, dtype=np.uint16)
        writer = StreamingPyramidWriter(f, levels, volume.shape)
        for z0 in range(0, nz, slab_z):
            z1 = min(z0 + slab_z, nz)
            slab = volume[z0:z1].copy()  # emulate the exporter's in-memory slab
            ds[z0:z1] = slab
            writer.consume(slab, z0)
        writer.finish()
        return {name: f[name][()] for name, *_ in levels if name in f}


def _check(volume, levels, slab_zs, label) -> None:
    tmp = Path(tempfile.mkdtemp())
    ref = _reference(volume, levels, tmp / "ref.h5")
    for slab_z in slab_zs:
        got = _streamed(volume, levels, tmp / f"got{slab_z}.h5", slab_z)
        assert set(got) == set(ref), f"{label}: level mismatch {set(got)} vs {set(ref)}"
        for name in ref:
            assert got[name].shape == ref[name].shape, (
                f"{label} slab_z={slab_z} {name}: shape {got[name].shape} != {ref[name].shape}"
            )
            assert np.array_equal(got[name], ref[name]), (
                f"{label} slab_z={slab_z} {name}: values differ "
                f"(max |diff| = {np.abs(got[name].astype(int) - ref[name].astype(int)).max()})"
            )


def test_block_sum_matches_float_average() -> None:
    """Integer block sums reproduce the original float64 mean exactly."""
    rng = np.random.default_rng(0)
    for shape, (fw, fh, fd) in (
        ((2, 8, 8), (2, 2, 2)), ((3, 9, 9), (3, 3, 3)), ((4, 12, 10), (2, 3, 4)),
    ):
        slab = rng.integers(0, 65536, size=shape, dtype=np.uint16)
        expect = (
            slab[:fd].astype(np.float64)
            .reshape(fd, shape[1] // fh, fh, shape[2] // fw, fw)
            .mean(axis=(0, 2, 4))
            .astype(np.uint16)
        )
        assert np.array_equal(block_average_3d(slab, fw, fh, fd), expect)
        sums = block_sum_3d(slab[:fd], fw, fh, fd)
        assert np.array_equal((sums[0] // (fw * fh * fd)).astype(np.uint16), expect)


def test_sum_dtype_avoids_overflow() -> None:
    """The chosen accumulator never overflows for uint16 inputs."""
    for factors in ((2, 2, 2), (8, 8, 8), (64, 64, 64)):
        dt = pyramid_sum_dtype(*factors)
        assert 65535 * factors[0] * factors[1] * factors[2] <= np.iinfo(dt).max


def test_hierarchical_ladder_matches_reference() -> None:
    """2/4/8 (each divides the next) must match the reread implementation."""
    rng = np.random.default_rng(1)
    vol = rng.integers(0, 65536, size=(40, 32, 24), dtype=np.uint16)
    levels = [("Data_2_2_2", 2, 2, 2), ("Data_4_4_4", 4, 4, 4), ("Data_8_8_8", 8, 8, 8)]
    _check(vol, levels, [1, 3, 8, 16, 40, 64], "hierarchical")


def test_non_divisible_ladder_matches_reference() -> None:
    """2 and 3 do not divide each other; both must still be exact."""
    rng = np.random.default_rng(2)
    vol = rng.integers(0, 65536, size=(30, 27, 18), dtype=np.uint16)
    levels = [("Data_2_2_2", 2, 2, 2), ("Data_3_3_3", 3, 3, 3)]
    _check(vol, levels, [1, 4, 7, 30], "non-divisible")


def test_odd_shapes_trim_like_reference() -> None:
    """Shapes not divisible by the factors must trim identically."""
    rng = np.random.default_rng(3)
    vol = rng.integers(0, 65536, size=(37, 29, 23), dtype=np.uint16)
    levels = [("Data_2_2_2", 2, 2, 2), ("Data_4_4_4", 4, 4, 4)]
    _check(vol, levels, [5, 9, 37], "odd shapes")


def test_anisotropic_factors_match_reference() -> None:
    """Independent X/Y/Z factors (Luxendo allows them) must match."""
    rng = np.random.default_rng(4)
    vol = rng.integers(0, 65536, size=(24, 30, 40), dtype=np.uint16)
    levels = [("Data_4_2_1", 4, 2, 1), ("Data_8_4_2", 8, 4, 2)]
    _check(vol, levels, [1, 6, 24], "anisotropic")


def test_zero_sized_level_is_skipped() -> None:
    """A level whose output would be empty is skipped, not written."""
    rng = np.random.default_rng(5)
    vol = rng.integers(0, 65536, size=(4, 8, 8), dtype=np.uint16)
    tmp = Path(tempfile.mkdtemp())
    levels = [("Data_2_2_2", 2, 2, 2), ("Data_64_64_64", 64, 64, 64)]
    with h5py.File(str(tmp / "o.h5"), "w") as f:
        f.create_dataset("Data", shape=vol.shape, dtype=np.uint16)
        w = StreamingPyramidWriter(f, levels, vol.shape)
        w.consume(vol, 0)
        info = w.finish()
        assert "Data_64_64_64" in info["skipped"]
        assert "Data_64_64_64" not in f
        assert "Data_2_2_2" in f


def test_all_levels_complete_and_bytes_reported() -> None:
    """finish() reports no gaps, and byte accounting matches the datasets."""
    rng = np.random.default_rng(6)
    vol = rng.integers(0, 65536, size=(33, 20, 20), dtype=np.uint16)
    levels = [("Data_2_2_2", 2, 2, 2), ("Data_4_4_4", 4, 4, 4)]
    tmp = Path(tempfile.mkdtemp())
    with h5py.File(str(tmp / "o.h5"), "w") as f:
        f.create_dataset("Data", shape=vol.shape, dtype=np.uint16)
        w = StreamingPyramidWriter(f, levels, vol.shape)
        total = 0
        for z0 in range(0, vol.shape[0], 7):
            z1 = min(z0 + 7, vol.shape[0])
            total += w.consume(vol[z0:z1].copy(), z0)
        info = w.finish()
        assert info["incomplete"] == [], info["incomplete"]
        expected = sum(
            f[name].size * 2 for name, *_ in levels
        )
        assert total == info["bytes_written"] == expected


def test_out_of_order_slab_rejected() -> None:
    """Slabs must arrive contiguously in Z; a gap is a programming error."""
    vol = np.zeros((8, 4, 4), dtype=np.uint16)
    tmp = Path(tempfile.mkdtemp())
    with h5py.File(str(tmp / "o.h5"), "w") as f:
        f.create_dataset("Data", shape=vol.shape, dtype=np.uint16)
        w = StreamingPyramidWriter(f, [("Data_2_2_2", 2, 2, 2)], vol.shape)
        w.consume(vol[:4], 0)
        try:
            w.consume(vol[4:], 6)  # wrong start
        except ValueError:
            return
        raise AssertionError("out-of-order slab was not rejected")


def test_export_never_rereads_output_for_pyramids() -> None:
    """The production H5 export must not call the reread-based generator.

    Pyramids come from the corrected slabs; if generate_pyramid_level is ever
    invoked during an export we have silently reintroduced a full reread of the
    output volume per level.
    """
    import json
    from unittest import mock

    from shifter import export_engine as ee
    from shifter.data_loader import H5Loader
    from shifter.h5_utils import H5FileManager
    from shifter.shift_manager import ShiftManager

    shape = (16, 16, 16)
    tmp = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(7)
    vol = rng.integers(0, 65536, size=shape, dtype=np.uint16)
    src = tmp / "ch0.lux.h5"
    with h5py.File(str(src), "w") as f:
        f.create_dataset("Data", data=vol, chunks=(8, 8, 8), dtype=np.uint16)
        f.create_dataset(
            "Data_2_2_2", shape=(8, 8, 8), dtype=np.uint16, chunks=(4, 4, 4)
        )
        f.create_dataset(
            "metadata", data=json.dumps({"processingInformation": {}}).encode()
        )

    mgr = H5FileManager()
    try:
        loaders = [H5Loader(src, mgr)]
        sm = ShiftManager()
        sm.init_channels([src.name], reference_index=0, colormaps=["green"])
        with mock.patch(
            "shifter.h5_utils.generate_pyramid_level"
        ) as reread:
            ee.run_export_h5(
                loaders, sm, tmp / "out", ram_percent=90, write_pyramids=True
            )
        reread.assert_not_called()
    finally:
        mgr.close_all()

    # ...and the level was still produced correctly.
    with h5py.File(str(tmp / "out" / src.name), "r") as f:
        assert f["Data_2_2_2"].shape == (8, 8, 8)
        expect = block_sum_3d(vol, 2, 2, 2) // 8
        assert np.array_equal(f["Data_2_2_2"][()], expect.astype(np.uint16))


def test_parallel_backend_is_bit_identical_to_numpy() -> None:
    """The numba XY reduction must equal the numpy one exactly.

    Parallelising is only safe because these are *integer* sums: addition is
    associative and commutative and the accumulator cannot overflow, so
    evaluation order is irrelevant. This would not hold for float means.
    """
    from shifter import h5_utils as hu

    if not hu._HAVE_NUMBA_PYRAMID:
        return  # numba not installed; nothing to compare

    rng = np.random.default_rng(11)
    original = hu._HAVE_NUMBA_PYRAMID
    try:
        for shape in ((7, 101, 97), (3, 64, 64), (33, 128, 130)):
            for fh, fw in ((2, 2), (3, 3), (4, 2), (1, 4), (5, 7)):
                for dtype in (np.uint16, np.uint32):
                    src = rng.integers(0, 65536, size=shape).astype(dtype)
                    sd = hu.pyramid_sum_dtype(fw, fh, 1)
                    hu._HAVE_NUMBA_PYRAMID = False
                    ref = hu._block_sum_xy(src, fh, fw, sd)
                    hu._HAVE_NUMBA_PYRAMID = True
                    got = hu._block_sum_xy(src, fh, fw, sd)
                    assert got.shape == ref.shape, (shape, fh, fw, dtype)
                    assert np.array_equal(got, ref), (
                        f"numba != numpy for shape={shape} factors=({fh},{fw}) {dtype}"
                    )
    finally:
        hu._HAVE_NUMBA_PYRAMID = original


def test_streaming_matches_reference_under_numba() -> None:
    """End-to-end streaming pyramids stay exact with the parallel backend on."""
    from shifter import h5_utils as hu

    if not hu._HAVE_NUMBA_PYRAMID:
        return
    rng = np.random.default_rng(12)
    vol = rng.integers(0, 65536, size=(37, 66, 70), dtype=np.uint16)
    levels = [("Data_2_2_2", 2, 2, 2), ("Data_4_4_4", 4, 4, 4), ("Data_3_3_3", 3, 3, 3)]
    _check(vol, levels, [4, 9, 37], "numba backend")


_TESTS = [
    test_parallel_backend_is_bit_identical_to_numpy,
    test_streaming_matches_reference_under_numba,
    test_export_never_rereads_output_for_pyramids,
    test_block_sum_matches_float_average,
    test_sum_dtype_avoids_overflow,
    test_hierarchical_ladder_matches_reference,
    test_non_divisible_ladder_matches_reference,
    test_odd_shapes_trim_like_reference,
    test_anisotropic_factors_match_reference,
    test_zero_sized_level_is_skipped,
    test_all_levels_complete_and_bytes_reported,
    test_out_of_order_slab_rejected,
]


def run_validation() -> bool:
    print("=" * 60)
    print("Streaming pyramid Validation")
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
