"""Validation for single-resolution companion-header reduction.

Builds synthetic multi-resolution Imaris (.ims) and BigDataViewer (*_bdv.h5)
headers whose data are external links to a .lux.h5, reduces them to a single
(full-resolution) level, and checks the result keeps only level 0 pointing at
``Data`` while preserving the rest of the header.

Runnable via pytest, or standalone::

    python -m shifter.tests.test_headers
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

from shifter.h5_utils import (
    reduce_header_to_single_resolution,
    write_roi_headers,
    write_single_resolution_headers,
)

_LEVEL_PATHS = ["Data", "Data_2_2_2", "Data_4_4_4"]


def _s1(text: object) -> np.ndarray:
    """Encode as an Imaris-style |S1 char-array attribute."""
    return np.frombuffer(str(text).encode("ascii"), dtype="S1").copy()


def _make_ims(
    path: Path,
    lux_name: str,
    levels: int = 3,
    dims: tuple[int, int, int] = (16, 16, 16),
    extent: list[tuple[float, float]] | None = None,
) -> None:
    with h5py.File(str(path), "w") as f:
        f.attrs["ImarisDataSet"] = "ImarisDataSet"
        ds = f.create_group("DataSet")
        for lvl in range(levels):
            grp = ds.create_group(f"ResolutionLevel {lvl}/TimePoint 0/Channel 0")
            grp["Data"] = h5py.ExternalLink(lux_name, _LEVEL_PATHS[lvl])
            grp.create_dataset("Histogram", data=np.zeros(256, dtype=np.uint64))
        img = f.create_group("DataSetInfo/Image")
        for name, value in zip(("X", "Y", "Z"), dims):
            img.attrs.create(name, _s1(value))
        if extent is not None:
            for i, (mn, mx) in enumerate(extent):
                img.attrs.create(f"ExtMin{i}", _s1(mn))
                img.attrs.create(f"ExtMax{i}", _s1(mx))
        f.create_dataset("Thumbnail/Data", data=np.zeros((4, 4), dtype=np.uint8))


def _make_bdv(path: Path, lux_name: str, levels: int = 3) -> None:
    with h5py.File(str(path), "w") as f:
        res = np.array([[2**i, 2**i, 2**i] for i in range(levels)], dtype=np.float64)
        sub = np.array([[64, 64, 64]] * levels, dtype=np.float64)
        f.create_dataset("s00/resolutions", data=res)
        f.create_dataset("s00/subdivisions", data=sub)
        for lvl in range(levels):
            f[f"t00000/s00/{lvl}/cells"] = h5py.ExternalLink(lux_name, _LEVEL_PATHS[lvl])


def test_reduce_ims_keeps_only_level0() -> None:
    tmp = Path(tempfile.mkdtemp())
    ims = tmp / "main_test.ims"
    _make_ims(ims, "chan.lux.h5")
    assert reduce_header_to_single_resolution(ims) is True

    with h5py.File(str(ims), "r") as f:
        levels = [k for k in f["DataSet"].keys() if k.startswith("ResolutionLevel")]
        assert levels == ["ResolutionLevel 0"], levels
        link = f["DataSet/ResolutionLevel 0/TimePoint 0/Channel 0"].get(
            "Data", getlink=True
        )
        assert isinstance(link, h5py.ExternalLink)
        assert link.path == "Data"
        assert link.filename == "chan.lux.h5"
        # Ancillary structure preserved.
        assert "DataSetInfo" in f and "Thumbnail" in f
        assert "Histogram" in f["DataSet/ResolutionLevel 0/TimePoint 0/Channel 0"]


def test_reduce_bdv_keeps_only_level0() -> None:
    tmp = Path(tempfile.mkdtemp())
    bdv = tmp / "main_test_bdv.h5"
    _make_bdv(bdv, "chan.lux.h5")
    assert reduce_header_to_single_resolution(bdv) is True

    with h5py.File(str(bdv), "r") as f:
        assert f["s00/resolutions"].shape == (1, 3)
        assert f["s00/resolutions"][:].tolist() == [[1.0, 1.0, 1.0]]
        assert f["s00/subdivisions"].shape == (1, 3)
        assert list(f["t00000/s00"].keys()) == ["0"]
        link = f["t00000/s00/0"].get("cells", getlink=True)
        assert isinstance(link, h5py.ExternalLink)
        assert link.path == "Data"


def test_reduce_is_idempotent() -> None:
    tmp = Path(tempfile.mkdtemp())
    ims = tmp / "main_test.ims"
    _make_ims(ims, "chan.lux.h5")
    reduce_header_to_single_resolution(ims)
    # Second pass must not error and must leave a single level.
    reduce_header_to_single_resolution(ims)
    with h5py.File(str(ims), "r") as f:
        levels = [k for k in f["DataSet"].keys() if k.startswith("ResolutionLevel")]
        assert levels == ["ResolutionLevel 0"]


def test_xml_left_unrecognized() -> None:
    tmp = Path(tempfile.mkdtemp())
    xml = tmp / "main_test_bdv.xml"
    xml.write_text("<SpimData></SpimData>")
    # Not an .ims / _bdv.h5 -> not reduced (returns False), content untouched.
    assert reduce_header_to_single_resolution(xml) is False
    assert xml.read_text() == "<SpimData></SpimData>"


def test_write_single_resolution_headers_copies_and_reduces() -> None:
    src_dir = Path(tempfile.mkdtemp())
    out_dir = Path(tempfile.mkdtemp())
    ims = src_dir / "main_test.ims"
    bdv = src_dir / "main_test_bdv.h5"
    xml = src_dir / "main_test_bdv.xml"
    _make_ims(ims, "chan.lux.h5")
    _make_bdv(bdv, "chan.lux.h5")
    xml.write_text("<SpimData/>")

    written = write_single_resolution_headers([ims, bdv, xml], out_dir)
    assert {p.name for p in written} == {ims.name, bdv.name, xml.name}

    # Copies reduced; originals untouched (still 3 levels).
    with h5py.File(str(out_dir / ims.name), "r") as f:
        assert [k for k in f["DataSet"].keys() if k.startswith("Resolution")] == [
            "ResolutionLevel 0"
        ]
    with h5py.File(str(src_dir / ims.name), "r") as f:
        assert len([k for k in f["DataSet"].keys() if k.startswith("Resolution")]) == 3
    with h5py.File(str(out_dir / bdv.name), "r") as f:
        assert f["s00/resolutions"].shape == (1, 3)
    # xml copied verbatim.
    assert (out_dir / xml.name).read_text() == "<SpimData/>"


def _read_ims_attr(grp, key: str) -> str:
    return grp.attrs[key].tobytes().decode("ascii")


def test_write_roi_headers_ims_crops_and_remaps() -> None:
    src = Path(tempfile.mkdtemp())
    out = Path(tempfile.mkdtemp())
    ims = src / "main_test.ims"
    # Full dims 100x80x60; extents chosen so voxel = 1, 2, 3 um on X, Y, Z.
    _make_ims(ims, "chan.lux.h5", levels=3, dims=(100, 80, 60),
              extent=[(0.0, 100.0), (0.0, 160.0), (0.0, 180.0)])
    roi = (10, 40, 20, 60, 5, 55)  # z0,z1, y0,y1, x0,x1

    written = write_roi_headers([ims], out, "_corrected_roi", roi, write_pyramids=False)
    assert [p.name for p in written] == [ims.name]

    with h5py.File(str(out / ims.name), "r") as f:
        # Single level, external link remapped to the ROI output filename.
        assert [k for k in f["DataSet"].keys() if k.startswith("Resolution")] == [
            "ResolutionLevel 0"
        ]
        link = f["DataSet/ResolutionLevel 0/TimePoint 0/Channel 0"].get(
            "Data", getlink=True
        )
        assert link.filename == "chan_corrected_roi.lux.h5"
        assert link.path == "Data"
        img = f["DataSetInfo/Image"]
        assert _read_ims_attr(img, "X") == "50"
        assert _read_ims_attr(img, "Y") == "40"
        assert _read_ims_attr(img, "Z") == "30"
        # Cropped extent, voxel size preserved (vox = 1, 2, 3).
        assert float(_read_ims_attr(img, "ExtMin0")) == 5.0
        assert float(_read_ims_attr(img, "ExtMax0")) == 55.0
        assert float(_read_ims_attr(img, "ExtMin1")) == 40.0
        assert float(_read_ims_attr(img, "ExtMax1")) == 120.0
        assert float(_read_ims_attr(img, "ExtMin2")) == 30.0
        assert float(_read_ims_attr(img, "ExtMax2")) == 120.0


def test_write_roi_headers_ims_keeps_levels_with_pyramids() -> None:
    src = Path(tempfile.mkdtemp())
    out = Path(tempfile.mkdtemp())
    ims = src / "main_test.ims"
    _make_ims(ims, "chan.lux.h5", levels=3, dims=(100, 80, 60),
              extent=[(0.0, 100.0), (0.0, 160.0), (0.0, 180.0)])
    write_roi_headers([ims], out, "_corrected_roi", (0, 30, 0, 40, 0, 50),
                      write_pyramids=True)
    with h5py.File(str(out / ims.name), "r") as f:
        levels = sorted(k for k in f["DataSet"].keys() if k.startswith("Resolution"))
        assert levels == ["ResolutionLevel 0", "ResolutionLevel 1", "ResolutionLevel 2"]
        # Every level's link is remapped to the ROI file, keeping its own path.
        for lvl, path in zip(levels, _LEVEL_PATHS):
            link = f[f"DataSet/{lvl}/TimePoint 0/Channel 0"].get("Data", getlink=True)
            assert link.filename == "chan_corrected_roi.lux.h5"
            assert link.path == path


def test_write_roi_headers_bdv_remaps() -> None:
    src = Path(tempfile.mkdtemp())
    out = Path(tempfile.mkdtemp())
    bdv = src / "main_test_bdv.h5"
    _make_bdv(bdv, "chan.lux.h5", levels=3)
    write_roi_headers([bdv], out, "_corrected_roi", (0, 4, 0, 8, 0, 8),
                      write_pyramids=False)
    with h5py.File(str(out / bdv.name), "r") as f:
        assert list(f["t00000/s00"].keys()) == ["0"]
        link = f["t00000/s00/0"].get("cells", getlink=True)
        assert link.filename == "chan_corrected_roi.lux.h5"
        assert link.path == "Data"


def test_write_roi_headers_skips_xml() -> None:
    src = Path(tempfile.mkdtemp())
    out = Path(tempfile.mkdtemp())
    xml = src / "main_test_bdv.xml"
    xml.write_text("<SpimData/>")
    written = write_roi_headers([xml], out, "_corrected_roi", (0, 1, 0, 1, 0, 1),
                                write_pyramids=False)
    assert written == []
    assert not (out / xml.name).exists()


_TESTS = [
    test_reduce_ims_keeps_only_level0,
    test_reduce_bdv_keeps_only_level0,
    test_reduce_is_idempotent,
    test_xml_left_unrecognized,
    test_write_single_resolution_headers_copies_and_reduces,
    test_write_roi_headers_ims_crops_and_remaps,
    test_write_roi_headers_ims_keeps_levels_with_pyramids,
    test_write_roi_headers_bdv_remaps,
    test_write_roi_headers_skips_xml,
]


def run_validation() -> bool:
    print("=" * 60)
    print("Single-resolution header Validation")
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
