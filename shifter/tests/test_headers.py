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
    rebuild_ims_header,
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


def _make_multires_ims_with_targets(d: Path, shape_zyx=(80, 60, 40)) -> tuple[Path, str]:
    """Build an .ims with uint8 attrs plus a REAL .lux.h5 target it links to."""
    nz, ny, nx = shape_zyx
    lux = d / "chan.lux.h5"
    with h5py.File(str(lux), "w") as f:
        f.create_dataset("Data", data=np.zeros((nz, ny, nx), np.uint16))
        f.create_dataset(
            "Data_2_2_2", data=np.zeros((nz // 2, ny // 2, nx // 2), np.uint16)
        )
    ims = d / "main_t.ims"
    with h5py.File(str(ims), "w") as f:
        for k, v in (
            ("DataSetDirectoryName", "DataSet"),
            ("DataSetInfoDirectoryName", "DataSetInfo"),
            ("ImarisDataSet", "ImarisDataSet"),
            ("ImarisVersion", "5.5.0"),
            ("ThumbnailDirectoryName", "Thumbnail"),
        ):
            f.attrs.create(k, np.frombuffer(v.encode(), dtype=np.uint8).copy())
        for lvl, (path, div) in enumerate((("Data", 1), ("Data_2_2_2", 2))):
            g = f.create_group(f"DataSet/ResolutionLevel {lvl}/TimePoint 0/Channel 0")
            g["Data"] = h5py.ExternalLink(lux.name, path)
            g.create_dataset("Histogram", data=np.zeros(256, np.uint64))
            for a, val in (("ImageSizeX", nx // div), ("ImageSizeY", ny // div),
                           ("ImageSizeZ", nz // div)):
                g.attrs.create(a, np.frombuffer(str(val).encode(), dtype=np.uint8).copy())
        img = f.create_group("DataSetInfo/Image")
        for a, val in (("X", nx), ("Y", ny), ("Z", nz), ("Unit", "um"),
                       ("ExtMin0", 0), ("ExtMax0", nx), ("ExtMin1", 0),
                       ("ExtMax1", ny), ("ExtMin2", 0), ("ExtMax2", nz)):
            img.attrs.create(a, np.frombuffer(str(val).encode(), dtype=np.uint8).copy())
        f.create_group("DataSetInfo/TimeInfo")
        # Imaris Viewer residue that must NOT survive a rebuild.
        f.create_dataset("Thumbnail/Data", data=np.zeros((4, 4), np.uint8))
        f.create_group("VolumeMask")
        f.create_group("DataSetInfo/Log")
    return ims, lux.name


def test_rebuilt_ims_matches_linked_shapes_and_is_uint8() -> None:
    """Every declared size must equal the shape of the dataset it links to."""
    d = Path(tempfile.mkdtemp())
    ims, _ = _make_multires_ims_with_targets(d)
    out = d / "out.ims"
    rebuild_ims_header(ims, out, output_shape_zyx=(80, 60, 40))

    with h5py.File(str(out), "r") as f:
        for lvl in f["DataSet"].keys():
            ch = f[f"DataSet/{lvl}/TimePoint 0/Channel 0"]
            # Dereference the link for real and compare to the declaration.
            linked = ch["Data"]
            declared = tuple(
                int(ch.attrs[k].tobytes().decode())
                for k in ("ImageSizeZ", "ImageSizeY", "ImageSizeX")
            )
            assert linked.shape == declared, f"{lvl}: {linked.shape} != {declared}"
            for k in ("ImageSizeX", "ImageSizeY", "ImageSizeZ"):
                assert ch.attrs[k].dtype == np.uint8, f"{lvl}/{k} not uint8"
        img = f["DataSetInfo/Image"]
        for k in ("X", "Y", "Z"):
            assert img.attrs[k].dtype == np.uint8


def test_rebuilt_ims_drops_imaris_viewer_residue() -> None:
    """A rebuild must emit only DataSet + DataSetInfo{Channel,Image,TimeInfo}."""
    d = Path(tempfile.mkdtemp())
    ims, _ = _make_multires_ims_with_targets(d)
    out = d / "out.ims"
    rebuild_ims_header(ims, out, output_shape_zyx=(80, 60, 40))
    with h5py.File(str(out), "r") as f:
        assert sorted(f.keys()) == ["DataSet", "DataSetInfo"], sorted(f.keys())
        assert "Log" not in f["DataSetInfo"]
        assert "NumberOfDataSets" not in f.attrs


def test_rebuilt_ims_single_res_drops_pyramid_levels() -> None:
    """keep_paths={'Data'} must leave exactly one level, linked to Data."""
    d = Path(tempfile.mkdtemp())
    ims, lux_name = _make_multires_ims_with_targets(d)
    out = d / "out.ims"
    info = rebuild_ims_header(
        ims, out, output_shape_zyx=(80, 60, 40), keep_paths={"Data"}
    )
    assert info["levels"] == ["ResolutionLevel 0"]
    with h5py.File(str(out), "r") as f:
        assert list(f["DataSet"].keys()) == ["ResolutionLevel 0"]
        ch = f["DataSet/ResolutionLevel 0/TimePoint 0/Channel 0"]
        assert ch["Data"].shape == (80, 60, 40)  # dereferences to the real target
        assert ch.get("Data", getlink=True).filename == lux_name


def test_rebuilt_roi_ims_declares_crop_everywhere() -> None:
    """A cropped rebuild must update per-level sizes too, not just Image X/Y/Z."""
    d = Path(tempfile.mkdtemp())
    ims, _ = _make_multires_ims_with_targets(d)
    out = d / "out.ims"
    roi = (10, 50, 5, 45, 0, 20)  # 40 x 40 x 20 (Z,Y,X)
    rebuild_ims_header(ims, out, output_shape_zyx=(40, 40, 20), roi=roi,
                       keep_paths={"Data"})
    with h5py.File(str(out), "r") as f:
        ch = f["DataSet/ResolutionLevel 0/TimePoint 0/Channel 0"]
        assert [int(ch.attrs[k].tobytes().decode())
                for k in ("ImageSizeX", "ImageSizeY", "ImageSizeZ")] == [20, 40, 40]
        img = f["DataSetInfo/Image"]
        assert [int(img.attrs[k].tobytes().decode()) for k in ("X", "Y", "Z")] == [
            20, 40, 40
        ]
        # voxel size preserved (source extent was 1 unit per voxel on each axis)
        assert float(img.attrs["ExtMin0"].tobytes().decode()) == 0.0
        assert float(img.attrs["ExtMax0"].tobytes().decode()) == 20.0
        assert float(img.attrs["ExtMin2"].tobytes().decode()) == 10.0
        assert float(img.attrs["ExtMax2"].tobytes().decode()) == 50.0


_TESTS = [
    test_rebuilt_ims_matches_linked_shapes_and_is_uint8,
    test_rebuilt_ims_drops_imaris_viewer_residue,
    test_rebuilt_ims_single_res_drops_pyramid_levels,
    test_rebuilt_roi_ims_declares_crop_everywhere,
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
