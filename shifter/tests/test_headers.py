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
    write_single_resolution_headers,
)

_LEVEL_PATHS = ["Data", "Data_2_2_2", "Data_4_4_4"]


def _make_ims(path: Path, lux_name: str, levels: int = 3) -> None:
    with h5py.File(str(path), "w") as f:
        f.attrs["ImarisDataSet"] = "ImarisDataSet"
        ds = f.create_group("DataSet")
        for lvl in range(levels):
            grp = ds.create_group(f"ResolutionLevel {lvl}/TimePoint 0/Channel 0")
            grp["Data"] = h5py.ExternalLink(lux_name, _LEVEL_PATHS[lvl])
            grp.create_dataset("Histogram", data=np.zeros(256, dtype=np.uint64))
        img = f.create_group("DataSetInfo/Image")
        img.attrs["X"] = "16"
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


_TESTS = [
    test_reduce_ims_keeps_only_level0,
    test_reduce_bdv_keeps_only_level0,
    test_reduce_is_idempotent,
    test_xml_left_unrecognized,
    test_write_single_resolution_headers_copies_and_reduces,
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
