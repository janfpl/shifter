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
    BdvXmlError,
    ExternalLinkMap,
    quarantine_header,
    rebuild_ims_header,
    reduce_header_to_single_resolution,
    validate_bdv_xml,
    validate_header,
    validate_headers,
    write_bdv_xml,
    write_roi_headers,
    write_single_resolution_headers,
)

_LEVEL_PATHS = ["Data", "Data_2_2_2", "Data_4_4_4"]

# A SpimData document shaped like the ones BigDataViewer writes: the setup's
# calibration transform is listed LAST (it consumes raw voxel indices), which is
# the ordering the crop-offset translation has to slot into.
_BDV_XML_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<SpimData version="0.2">
  <BasePath type="relative">.</BasePath>
  <SequenceDescription>
    <ImageLoader format="bdv.hdf5">
      <hdf5 type="relative">{h5_name}</hdf5>
    </ImageLoader>
    <ViewSetups>
      <ViewSetup>
        <id>0</id>
        <name>channel 0</name>
        <size>{nx} {ny} {nz}</size>
        <voxelSize>
          <unit>micron</unit>
          <size>0.406 0.406 2.0</size>
        </voxelSize>
        <attributes>
          <illumination>0</illumination>
          <channel>0</channel>
          <tile>0</tile>
          <angle>0</angle>
        </attributes>
      </ViewSetup>
      <Attributes name="channel">
        <Channel><id>0</id><name>0</name></Channel>
      </Attributes>
    </ViewSetups>
    <Timepoints type="range">
      <first>0</first>
      <last>0</last>
    </Timepoints>
    <MissingViews />
  </SequenceDescription>
  <ViewRegistrations>
    <ViewRegistration timepoint="0" setup="0">
      <ViewTransform type="affine">
        <name>calibration</name>
        <affine>0.406 0.0 0.0 100.0 0.0 0.406 0.0 200.0 0.0 0.0 2.0 300.0</affine>
      </ViewTransform>
    </ViewRegistration>
  </ViewRegistrations>
</SpimData>
"""


def _make_bdv_xml(
    path: Path, h5_name: str, shape_zyx: tuple[int, int, int] = (16, 16, 16)
) -> None:
    nz, ny, nx = shape_zyx
    path.write_text(
        _BDV_XML_TEMPLATE.format(h5_name=h5_name, nx=nx, ny=ny, nz=nz)
    )


def _bdv_model(xml_path: Path, setup: str = "0") -> np.ndarray:
    """Compose a ViewRegistration's transforms the way SpimData does."""
    import xml.etree.ElementTree as ET

    root = ET.parse(str(xml_path)).getroot()
    model = np.eye(4)
    for reg in root.findall("./ViewRegistrations/ViewRegistration"):
        if reg.get("setup") != setup:
            continue
        for transform in reg.findall("ViewTransform"):
            m = np.eye(4)
            m[:3, :] = np.array(
                [float(v) for v in transform.find("affine").text.split()]
            ).reshape(3, 4)
            model = model @ m
    return model


def _xml_sizes(xml_path: Path) -> list[tuple[int, int, int]]:
    import xml.etree.ElementTree as ET

    root = ET.parse(str(xml_path)).getroot()
    return [
        tuple(int(v) for v in setup.find("size").text.split())
        for setup in root.findall("./SequenceDescription/ViewSetups/ViewSetup")
    ]


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
    _make_bdv_xml(xml, bdv.name)

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
    # The XML still describes the full volume and still points at its H5.
    assert _xml_sizes(out_dir / xml.name) == [(16, 16, 16)]
    assert validate_bdv_xml(out_dir / xml.name) == []


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
    xml = src / "main_test_bdv.xml"
    _make_bdv(bdv, "chan.lux.h5", levels=3)
    _make_bdv_xml(xml, bdv.name)
    write_roi_headers([bdv, xml], out, "_corrected_roi", (0, 4, 0, 8, 0, 8),
                      write_pyramids=False)
    with h5py.File(str(out / bdv.name), "r") as f:
        assert list(f["t00000/s00"].keys()) == ["0"]
        link = f["t00000/s00/0"].get("cells", getlink=True)
        assert link.filename == "chan_corrected_roi.lux.h5"
        assert link.path == "Data"


# --------------------------------------------------------------------------- #
# BigDataViewer XML
# --------------------------------------------------------------------------- #


def test_write_roi_headers_writes_xml_with_crop_size_and_offset() -> None:
    """A ROI export must produce a BDV XML declaring the crop, placed correctly."""
    src = Path(tempfile.mkdtemp())
    out = Path(tempfile.mkdtemp())
    bdv = src / "main_test_bdv.h5"
    xml = src / "main_test_bdv.xml"
    _make_bdv(bdv, "chan.lux.h5", levels=3)
    _make_bdv_xml(xml, bdv.name, shape_zyx=(16, 16, 16))

    roi = (2, 10, 3, 11, 4, 12)  # z0,z1, y0,y1, x0,x1 -> 8 x 8 x 8
    written = write_roi_headers([bdv, xml], out, "_corrected_roi", roi,
                                write_pyramids=False, input_shape_zyx=(16, 16, 16))
    assert {p.name for p in written} == {bdv.name, xml.name}

    dst_xml = out / xml.name
    assert _xml_sizes(dst_xml) == [(8, 8, 8)]
    assert validate_bdv_xml(dst_xml, expected_shape_zyx=(8, 8, 8)) == []

    # Voxel (0,0,0) of the crop must land where voxel (4,3,2) of the full
    # volume did — i.e. the crop keeps its position inside the specimen.
    full_model = _bdv_model(xml)
    crop_model = _bdv_model(dst_xml)
    crop_origin = crop_model @ np.array([0.0, 0.0, 0.0, 1.0])
    expected = full_model @ np.array([4.0, 3.0, 2.0, 1.0])
    assert np.allclose(crop_origin, expected), (crop_origin, expected)
    # The linear part is untouched: a crop resamples nothing.
    assert np.allclose(crop_model[:3, :3], full_model[:3, :3])


def test_roi_xml_keeps_voxel_size_and_timepoints() -> None:
    import xml.etree.ElementTree as ET

    src = Path(tempfile.mkdtemp())
    out = Path(tempfile.mkdtemp())
    xml = src / "main_test_bdv.xml"
    _make_bdv_xml(xml, "main_test_bdv.h5")
    write_bdv_xml(xml, out / xml.name, output_shape_zyx=(4, 4, 4),
                  roi=(0, 4, 0, 4, 0, 4), h5_filename="main_test_bdv.h5")

    root = ET.parse(str(out / xml.name)).getroot()
    setup = root.find("./SequenceDescription/ViewSetups/ViewSetup")
    assert setup.find("voxelSize/size").text == "0.406 0.406 2.0"
    assert setup.find("attributes/channel").text == "0"
    tps = root.find("./SequenceDescription/Timepoints")
    assert (tps.find("first").text, tps.find("last").text) == ("0", "0")


def test_bdv_xml_repoints_nested_h5_reference_to_flat_output() -> None:
    """A nested <hdf5> path must not keep pointing outside a flat output folder."""
    src = Path(tempfile.mkdtemp())
    out = Path(tempfile.mkdtemp())
    xml = src / "main_test_bdv.xml"
    _make_bdv_xml(xml, "raw/stack_1/main_test_bdv.h5")
    write_bdv_xml(xml, out / xml.name, h5_filename="main_test_bdv.h5")

    import xml.etree.ElementTree as ET

    root = ET.parse(str(out / xml.name)).getroot()
    element = root.find("./SequenceDescription/ImageLoader/hdf5")
    assert element.text == "main_test_bdv.h5"
    assert element.get("type") == "relative"


def test_bdv_xml_rejects_setup_of_different_size() -> None:
    src = Path(tempfile.mkdtemp())
    out = Path(tempfile.mkdtemp())
    xml = src / "main_test_bdv.xml"
    _make_bdv_xml(xml, "main_test_bdv.h5", shape_zyx=(16, 16, 16))
    try:
        write_bdv_xml(xml, out / xml.name, output_shape_zyx=(8, 8, 8),
                      input_shape_zyx=(32, 32, 32), roi=(0, 8, 0, 8, 0, 8))
    except BdvXmlError:
        pass
    else:
        raise AssertionError("expected BdvXmlError for a mismatched ViewSetup size")


def test_bdv_pair_is_complete_or_wholly_absent() -> None:
    """An unusable XML must take the H5 down with it — never leave half a pair."""
    src = Path(tempfile.mkdtemp())
    out = Path(tempfile.mkdtemp())
    bdv = src / "main_test_bdv.h5"
    xml = src / "main_test_bdv.xml"
    _make_bdv(bdv, "chan.lux.h5", levels=3)
    xml.write_text("<SpimData/>")  # parses, but has no ViewSetups

    written = write_roi_headers([bdv, xml], out, "_corrected_roi",
                                (0, 4, 0, 8, 0, 8), write_pyramids=False)
    assert written == []
    assert not (out / bdv.name).exists()
    assert not (out / xml.name).exists()


def test_bdv_h5_without_xml_is_omitted() -> None:
    """BigDataViewer opens the XML, so an H5 alone is a dataset nobody can open."""
    src = Path(tempfile.mkdtemp())
    out = Path(tempfile.mkdtemp())
    bdv = src / "main_test_bdv.h5"
    _make_bdv(bdv, "chan.lux.h5", levels=3)
    written = write_single_resolution_headers([bdv], out)
    assert written == []
    assert not (out / bdv.name).exists()


# --------------------------------------------------------------------------- #
# Real manufacturer sample
#
# tests/data/main_st-0-x00-y00-0-x00-y01_bdv.xml is a genuine Luxendo BDV XML
# for a 3099 x 6979 x 5347 two-channel acquisition. Synthetic fixtures only
# prove the code agrees with itself; this one is the check that it agrees with
# what the microscope actually writes.
# --------------------------------------------------------------------------- #

_SAMPLE_XML = Path(__file__).parent / "data" / "main_st-0-x00-y00-0-x00-y01_bdv.xml"
_SAMPLE_SHAPE = (3099, 6979, 5347)  # (nz, ny, nx), i.e. <size>5347 6979 3099</size>


def test_real_bdv_xml_full_resolution_is_unchanged_but_repointed() -> None:
    """A full-volume export must leave sizes and registrations exactly as found."""
    out = Path(tempfile.mkdtemp())
    dst = out / _SAMPLE_XML.name
    write_bdv_xml(_SAMPLE_XML, dst, output_shape_zyx=_SAMPLE_SHAPE,
                  input_shape_zyx=_SAMPLE_SHAPE, h5_filename="renamed_bdv.h5")

    assert _xml_sizes(dst) == [(5347, 6979, 3099), (5347, 6979, 3099)]
    for setup in ("0", "1"):
        assert np.allclose(_bdv_model(dst, setup), _bdv_model(_SAMPLE_XML, setup))

    import xml.etree.ElementTree as ET

    root = ET.parse(str(dst)).getroot()
    assert root.find("./SequenceDescription/ImageLoader/hdf5").text == "renamed_bdv.h5"
    # Everything the rewrite does not understand is carried through untouched.
    assert root.find("BasePath").text == "."
    assert [a.get("name") for a in root.findall(".//Attributes")] == [
        "channel", "angle", "tile"
    ]
    setups = root.findall(".//ViewSetup")
    assert [s.find("voxelSize/size").text for s in setups] == [
        "2.925 2.925 3", "2.925 2.925 3"
    ]
    assert [s.find("attributes/channel").text for s in setups] == ["0", "1"]
    assert setups[0].find("name").text.startswith("ch:0_st:0-x00-y00")


def test_real_bdv_xml_crop_lands_at_the_right_micrometre() -> None:
    """The crop's world origin must shift by the offset *scaled by calibration*.

    The sample's registration is the calibration itself (2.925, 2.925, 3 um per
    voxel plus a translation), so this pins down the transform ordering: the
    crop offset is in voxels and must be consumed by that calibration. Applying
    it on the wrong side would displace the crop by ~2 mm — a plausible-looking
    volume in the wrong place, which is exactly the failure worth a test.
    """
    out = Path(tempfile.mkdtemp())
    dst = out / _SAMPLE_XML.name
    roi = (500, 1500, 2000, 4000, 1000, 3000)  # z0,z1, y0,y1, x0,x1
    write_bdv_xml(_SAMPLE_XML, dst, output_shape_zyx=(1000, 2000, 2000),
                  input_shape_zyx=_SAMPLE_SHAPE, roi=roi,
                  h5_filename="main_st-0-x00-y00-0-x00-y01_bdv.h5")

    assert _xml_sizes(dst) == [(2000, 2000, 1000), (2000, 2000, 1000)]
    for setup in ("0", "1"):
        full, crop = _bdv_model(_SAMPLE_XML, setup), _bdv_model(dst, setup)
        origin = crop @ np.array([0.0, 0.0, 0.0, 1.0])
        # -9006.575 + 2.925*1000, -10799.059 + 2.925*2000, -8872.391 + 3*500
        assert np.allclose(origin[:3], [-6081.575, -4949.059, -7372.391]), origin
        # ... and every other voxel follows the same mapping.
        for v in ((0, 0, 0), (500, 700, 900), (1999, 1999, 999)):
            got = crop @ np.array([*v, 1.0])
            want = full @ np.array([v[0] + 1000, v[1] + 2000, v[2] + 500, 1.0])
            assert np.allclose(got, want), (setup, v, got, want)
        assert np.allclose(crop[:3, :3], full[:3, :3])


def test_real_bdv_xml_validates_against_its_h5() -> None:
    """Both ViewSetups must resolve to setup groups in the paired H5."""
    out = Path(tempfile.mkdtemp())
    dst = out / _SAMPLE_XML.name
    h5_name = "main_st-0-x00-y00-0-x00-y01_bdv.h5"
    write_bdv_xml(_SAMPLE_XML, dst, output_shape_zyx=_SAMPLE_SHAPE,
                  input_shape_zyx=_SAMPLE_SHAPE, h5_filename=h5_name)
    with h5py.File(str(out / h5_name), "w") as f:
        for s in ("s00", "s01"):
            f.create_dataset(f"{s}/resolutions", data=np.ones((1, 3)))
            f.create_dataset(f"{s}/subdivisions", data=np.full((1, 3), 64.0))
    assert validate_bdv_xml(dst, expected_shape_zyx=_SAMPLE_SHAPE) == []

    # A setup with no matching sNN group is a dataset BigDataViewer cannot open.
    (out / h5_name).unlink()
    with h5py.File(str(out / h5_name), "w") as f:
        f.create_dataset("s00/resolutions", data=np.ones((1, 3)))
    problems = validate_bdv_xml(dst, expected_shape_zyx=_SAMPLE_SHAPE)
    assert len(problems) == 1 and "ViewSetup 1" in problems[0], problems


def test_validate_bdv_xml_flags_missing_h5_and_wrong_size() -> None:
    d = Path(tempfile.mkdtemp())
    xml = d / "main_test_bdv.xml"
    _make_bdv_xml(xml, "absent_bdv.h5", shape_zyx=(16, 16, 16))
    problems = validate_bdv_xml(xml, expected_shape_zyx=(8, 8, 8))
    assert any("does not exist" in p for p in problems), problems
    assert any("declares size" in p for p in problems), problems


def test_validate_bdv_xml_flags_setup_without_h5_group() -> None:
    d = Path(tempfile.mkdtemp())
    bdv = d / "main_test_bdv.h5"
    xml = d / "main_test_bdv.xml"
    with h5py.File(str(bdv), "w") as f:  # setup s01, but the XML declares id 0
        f.create_dataset("s01/resolutions", data=np.ones((1, 3)))
    _make_bdv_xml(xml, bdv.name)
    problems = validate_bdv_xml(xml)
    assert any("no matching setup group" in p for p in problems), problems


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


# --------------------------------------------------------------------------- #
# Exact source -> output link mapping
#
# These fixtures build REAL .lux.h5 targets, so every assertion dereferences the
# links the way a viewer does. A test that only inspects the ExternalLink object
# would pass just as happily on a header pointing at nothing — or at the
# uncorrected source data, which is the failure worth catching.
# --------------------------------------------------------------------------- #

_DATA_SHAPE = (8, 8, 8)


def _write_lux(path: Path, shape_zyx: tuple[int, int, int], pyramids: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nz, ny, nx = shape_zyx
    with h5py.File(str(path), "w") as f:
        f.create_dataset("Data", data=np.zeros((nz, ny, nx), np.uint16))
        if pyramids:
            f.create_dataset(
                "Data_2_2_2", data=np.zeros((nz // 2, ny // 2, nx // 2), np.uint16)
            )


def _u8(value: object) -> np.ndarray:
    return np.frombuffer(str(value).encode(), dtype=np.uint8).copy()


def _make_acquisition(
    root: Path,
    *,
    nested: bool = False,
    n_channels: int = 2,
    pyramids: bool = True,
    shape_zyx: tuple[int, int, int] = _DATA_SHAPE,
    same_basename: bool = False,
) -> dict:
    """Build an acquisition: real per-channel data plus .ims and BDV headers.

    With *nested*, each channel lives in its own subfolder and the headers link
    to it by a relative path — the layout that breaks a filename-based rule.
    With *same_basename*, every channel file is called ``Cam_left_00000.lux.h5``,
    which is what real nested acquisitions actually look like.
    """
    nz, ny, nx = shape_zyx
    root.mkdir(parents=True, exist_ok=True)

    sources: list[Path] = []
    links: list[str] = []
    for ch in range(n_channels):
        if nested:
            name = "Cam_left_00000.lux.h5" if same_basename else f"Cam_left_{ch}.lux.h5"
            rel = f"raw/stack_1_channel_{ch}-561_obj_left/{name}"
        else:
            rel = f"chan{ch}.lux.h5"
        path = root / rel
        _write_lux(path, shape_zyx, pyramids)
        sources.append(path)
        links.append(rel)

    level_paths = ["Data"] + (["Data_2_2_2"] if pyramids else [])

    ims = root / "main_test.ims"
    with h5py.File(str(ims), "w") as f:
        for key, value in (
            ("DataSetDirectoryName", "DataSet"),
            ("DataSetInfoDirectoryName", "DataSetInfo"),
            ("ImarisDataSet", "ImarisDataSet"),
            ("ImarisVersion", "5.5.0"),
            ("ThumbnailDirectoryName", "Thumbnail"),
        ):
            f.attrs.create(key, _u8(value))
        for lvl, path in enumerate(level_paths):
            div = 2 if path != "Data" else 1
            for ch in range(n_channels):
                g = f.create_group(
                    f"DataSet/ResolutionLevel {lvl}/TimePoint 0/Channel {ch}"
                )
                g["Data"] = h5py.ExternalLink(links[ch], path)
                g.create_dataset("Histogram", data=np.zeros(256, np.uint64))
                for attr, val in (("ImageSizeX", nx // div), ("ImageSizeY", ny // div),
                                  ("ImageSizeZ", nz // div)):
                    g.attrs.create(attr, _u8(val))
        img = f.create_group("DataSetInfo/Image")
        for attr, val in (("X", nx), ("Y", ny), ("Z", nz), ("Unit", "um"),
                          ("ExtMin0", 0), ("ExtMax0", nx), ("ExtMin1", 0),
                          ("ExtMax1", ny), ("ExtMin2", 0), ("ExtMax2", nz)):
            img.attrs.create(attr, _u8(val))
        f.create_group("DataSetInfo/TimeInfo")

    bdv_h5 = root / "main_test_bdv.h5"
    with h5py.File(str(bdv_h5), "w") as f:
        res = np.array([[2**i] * 3 for i in range(len(level_paths))], dtype=float)
        for ch in range(n_channels):
            f.create_dataset(f"s{ch:02d}/resolutions", data=res)
            f.create_dataset(
                f"s{ch:02d}/subdivisions", data=np.full((len(level_paths), 3), 8.0)
            )
            for lvl, path in enumerate(level_paths):
                f[f"t00000/s{ch:02d}/{lvl}/cells"] = h5py.ExternalLink(links[ch], path)

    bdv_xml = root / "main_test_bdv.xml"
    setups = "".join(
        f"<ViewSetup><id>{ch}</id><name>ch{ch}</name><size>{nx} {ny} {nz}</size>"
        "<voxelSize><unit>micrometer</unit><size>1 1 1</size></voxelSize>"
        f"<attributes><channel>{ch}</channel></attributes></ViewSetup>"
        for ch in range(n_channels)
    )
    regs = "".join(
        f'<ViewRegistration timepoint="0" setup="{ch}"><ViewTransform type="affine">'
        "<affine>1 0 0 0 0 1 0 0 0 0 1 0</affine></ViewTransform></ViewRegistration>"
        for ch in range(n_channels)
    )
    bdv_xml.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<SpimData version="0.2"><BasePath type="relative">.</BasePath>'
        '<SequenceDescription><ImageLoader format="bdv.hdf5">'
        f'<hdf5 type="relative">{bdv_h5.name}</hdf5></ImageLoader>'
        f"<ViewSetups>{setups}</ViewSetups>"
        '<Timepoints type="range"><first>0</first><last>0</last></Timepoints>'
        f"</SequenceDescription><ViewRegistrations>{regs}</ViewRegistrations></SpimData>"
    )

    return {
        "dir": root, "ims": ims, "bdv_h5": bdv_h5, "bdv_xml": bdv_xml,
        "sources": sources, "headers": [ims, bdv_h5, bdv_xml],
    }


def _export_outputs(
    acq: dict, out_dir: Path, suffix: str, shape_zyx: tuple[int, int, int],
    pyramids: bool,
) -> ExternalLinkMap:
    """Write the corrected output files an export would, and record the mapping."""
    from shifter.utils import h5_output_filename

    out_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    for source in acq["sources"]:
        out_name = h5_output_filename(source.name, suffix)
        _write_lux(out_dir / out_name, shape_zyx, pyramids)
        mapping[str(source)] = out_name
    return ExternalLinkMap(mapping, header_dir=acq["dir"])


def _assert_links_resolve(header: Path) -> None:
    problems = validate_header(header)
    assert problems == [], problems


def test_nested_links_are_repointed_at_the_corrected_output() -> None:
    """The dangerous case: a nested link must not survive into a flat output.

    Left as-is, ``raw/stack_1_channel_0/Cam_left_00000.lux.h5`` either dangles or
    resolves back to the *uncorrected* source — the user then views raw data
    believing it is corrected.
    """
    root = Path(tempfile.mkdtemp())
    acq = _make_acquisition(root / "acq", nested=True, same_basename=False)
    out = root / "out"
    link_map = _export_outputs(acq, out, "", _DATA_SHAPE, pyramids=False)

    written = write_single_resolution_headers(
        acq["headers"], out, output_shape_zyx=_DATA_SHAPE, link_map=link_map
    )
    assert {p.name for p in written} == {"main_test.ims", "main_test_bdv.h5",
                                         "main_test_bdv.xml"}
    for path in written:
        _assert_links_resolve(path)

    # Flat filenames, and every link lands in the output folder.
    with h5py.File(str(out / "main_test.ims"), "r") as f:
        for ch in range(2):
            link = f[f"DataSet/ResolutionLevel 0/TimePoint 0/Channel {ch}"].get(
                "Data", getlink=True
            )
            assert "/" not in link.filename, link.filename
            assert (out / link.filename).is_file()
            assert not (root / "acq" / link.filename).exists()


def test_ambiguous_basenames_are_refused_not_guessed() -> None:
    """Two sources sharing a basename must not be resolved by name."""
    root = Path(tempfile.mkdtemp())
    acq = _make_acquisition(root / "acq", nested=True, same_basename=True)
    out = root / "out"

    # Both channels are Cam_left_00000.lux.h5, so the export must rename them;
    # a basename lookup can no longer say which one a link meant.
    out.mkdir()
    mapping = {}
    for ch, source in enumerate(acq["sources"]):
        out_name = f"Cam_left_00000_ch{ch}.lux.h5"
        _write_lux(out / out_name, _DATA_SHAPE, pyramids=False)
        mapping[str(source)] = out_name
    link_map = ExternalLinkMap(mapping, header_dir=acq["dir"])

    assert link_map.ambiguous_basenames == {"cam_left_00000.lux.h5"} or (
        link_map.ambiguous_basenames == {"Cam_left_00000.lux.h5"}
    ), link_map.ambiguous_basenames
    # A bare basename is refused...
    assert link_map.resolve("Cam_left_00000.lux.h5") is None
    assert "not unique" in link_map.explain("Cam_left_00000.lux.h5")
    # ...but the full nested path each header actually stores is unambiguous.
    assert link_map.resolve(
        "raw/stack_1_channel_1-561_obj_left/Cam_left_00000.lux.h5"
    ) == "Cam_left_00000_ch1.lux.h5"


def test_unresolvable_link_takes_the_header_down_with_it() -> None:
    """A link that cannot be mapped must not be left pointing at the source."""
    root = Path(tempfile.mkdtemp())
    acq = _make_acquisition(root / "acq", nested=False, n_channels=2)
    out = root / "out"
    # Only channel 0 was exported, so channel 1's link cannot be mapped.
    out.mkdir()
    _write_lux(out / "chan0.lux.h5", _DATA_SHAPE, pyramids=False)
    link_map = ExternalLinkMap({str(acq["sources"][0]): "chan0.lux.h5"},
                               header_dir=acq["dir"])

    written = write_single_resolution_headers(
        [acq["ims"]], out, output_shape_zyx=_DATA_SHAPE, link_map=link_map
    )
    assert written == []
    assert not (out / "main_test.ims").exists()


def test_link_map_lookup_orders() -> None:
    root = Path(tempfile.mkdtemp())
    acq = _make_acquisition(root / "acq", nested=True, n_channels=1)
    source = acq["sources"][0]
    link_map = ExternalLinkMap({str(source): "out.lux.h5"}, header_dir=acq["dir"])

    rel = "raw/stack_1_channel_0-561_obj_left/Cam_left_0.lux.h5"
    assert link_map.resolve(str(source)) == "out.lux.h5"          # exact
    assert link_map.resolve(rel) == "out.lux.h5"                  # relative to header
    assert link_map.resolve(rel.replace("/", "\\")) == "out.lux.h5"  # windows sep
    assert link_map.resolve("Cam_left_0.lux.h5") == "out.lux.h5"  # unambiguous base
    assert link_map.resolve("somewhere_else.lux.h5") is None
    assert "does not correspond" in link_map.explain("somewhere_else.lux.h5")


def _matrix_case(pyramids: bool, crop: bool) -> None:
    """Rebuild headers for one (pyramids, crop) combination and validate them."""
    root = Path(tempfile.mkdtemp())
    acq = _make_acquisition(root / "acq", nested=True, pyramids=pyramids)
    out = root / "out"

    if crop:
        roi = (2, 6, 2, 6, 2, 6)
        shape = (4, 4, 4)
        suffix = "_corrected_roi"
    else:
        roi = None
        shape = _DATA_SHAPE
        suffix = ""

    link_map = _export_outputs(acq, out, suffix, shape, pyramids=pyramids)
    if crop:
        written = write_roi_headers(
            acq["headers"], out, suffix, roi, pyramids,
            input_shape_zyx=_DATA_SHAPE, link_map=link_map,
        )
    else:
        written = write_single_resolution_headers(
            acq["headers"], out, output_shape_zyx=shape, link_map=link_map
        )

    assert len(written) == 3, [p.name for p in written]
    problems = validate_headers(written)
    assert problems == [], problems

    # Declared sizes must equal the ACTUAL shapes of the linked datasets.
    with h5py.File(str(out / "main_test.ims"), "r") as f:
        for level in f["DataSet"].keys():
            for ch in f[f"DataSet/{level}/TimePoint 0"].keys():
                group = f[f"DataSet/{level}/TimePoint 0/{ch}"]
                declared = tuple(
                    int(group.attrs[k].tobytes().decode())
                    for k in ("ImageSizeZ", "ImageSizeY", "ImageSizeX")
                )
                assert group["Data"].shape == declared, (level, ch, declared)


def test_matrix_pyramids_no_crop() -> None:
    _matrix_case(pyramids=True, crop=False)


def test_matrix_no_pyramids_no_crop() -> None:
    _matrix_case(pyramids=False, crop=False)


def test_matrix_pyramids_crop() -> None:
    _matrix_case(pyramids=True, crop=True)


def test_matrix_no_pyramids_crop() -> None:
    _matrix_case(pyramids=False, crop=True)


def test_validate_header_dereferences_rather_than_inspecting() -> None:
    """A link object that looks fine but leads nowhere must be reported."""
    d = Path(tempfile.mkdtemp())
    ims = d / "dangling.ims"
    with h5py.File(str(ims), "w") as f:
        g = f.create_group("DataSet/ResolutionLevel 0/TimePoint 0/Channel 0")
        g["Data"] = h5py.ExternalLink("absent.lux.h5", "Data")
        for attr, val in (("ImageSizeX", 8), ("ImageSizeY", 8), ("ImageSizeZ", 8)):
            g.attrs.create(attr, _u8(val))
        img = f.create_group("DataSetInfo/Image")
        for attr, val in (("X", 8), ("Y", 8), ("Z", 8)):
            img.attrs.create(attr, _u8(val))
    problems = validate_header(ims)
    assert any("does not exist" in p for p in problems), problems


def test_validate_header_catches_size_lie_and_s1_attrs() -> None:
    """Declared sizes must match the linked data, and text attrs must be uint8."""
    d = Path(tempfile.mkdtemp())
    _write_lux(d / "chan.lux.h5", (8, 8, 8), pyramids=False)
    ims = d / "wrong.ims"
    with h5py.File(str(ims), "w") as f:
        g = f.create_group("DataSet/ResolutionLevel 0/TimePoint 0/Channel 0")
        g["Data"] = h5py.ExternalLink("chan.lux.h5", "Data")
        g.attrs.create("ImageSizeX", _u8(999))          # a lie
        g.attrs.create("ImageSizeY", _u8(8))
        g.attrs.create("ImageSizeZ", _u8(8))
        img = f.create_group("DataSetInfo/Image")
        for attr in ("X", "Y", "Z"):
            img.attrs.create(attr, np.frombuffer(b"8", dtype="S1").copy())  # S1, not uint8
    problems = validate_header(ims)
    assert any("declares 999x8x8" in p for p in problems), problems
    assert any("expected a uint8 char array" in p for p in problems), problems


def test_validate_bdv_h5_catches_advertised_but_absent_level() -> None:
    """resolutions must not claim more levels than the file actually links to."""
    d = Path(tempfile.mkdtemp())
    _write_lux(d / "chan.lux.h5", (8, 8, 8), pyramids=True)
    bdv = d / "over_bdv.h5"
    with h5py.File(str(bdv), "w") as f:
        f.create_dataset("s00/resolutions", data=np.array([[1, 1, 1], [2, 2, 2]], float))
        f.create_dataset("s00/subdivisions", data=np.full((2, 3), 8.0))
        f["t00000/s00/0/cells"] = h5py.ExternalLink("chan.lux.h5", "Data")  # only one
    problems = validate_header(bdv)
    assert any("advertise a level that is not there" in p for p in problems), problems


def test_validate_bdv_h5_catches_factor_shape_mismatch() -> None:
    """A level's linked shape must equal full shape / its own downsample factors."""
    d = Path(tempfile.mkdtemp())
    with h5py.File(str(d / "chan.lux.h5"), "w") as f:
        f.create_dataset("Data", data=np.zeros((8, 8, 8), np.uint16))
        f.create_dataset("Data_2_2_2", data=np.zeros((8, 8, 8), np.uint16))  # not halved
    bdv = d / "bad_bdv.h5"
    with h5py.File(str(bdv), "w") as f:
        f.create_dataset("s00/resolutions", data=np.array([[1, 1, 1], [2, 2, 2]], float))
        f.create_dataset("s00/subdivisions", data=np.full((2, 3), 8.0))
        f["t00000/s00/0/cells"] = h5py.ExternalLink("chan.lux.h5", "Data")
        f["t00000/s00/1/cells"] = h5py.ExternalLink("chan.lux.h5", "Data_2_2_2")
    problems = validate_header(bdv)
    assert any("imply" in p for p in problems), problems


def test_quarantine_header_makes_it_unopenable() -> None:
    d = Path(tempfile.mkdtemp())
    bad = d / "main_test.ims"
    bad.write_bytes(b"not really a header")
    moved = quarantine_header(bad)
    assert moved is not None and moved.name == "main_test.ims.invalid"
    assert not bad.exists() and moved.is_file()


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
    test_write_roi_headers_writes_xml_with_crop_size_and_offset,
    test_roi_xml_keeps_voxel_size_and_timepoints,
    test_bdv_xml_repoints_nested_h5_reference_to_flat_output,
    test_bdv_xml_rejects_setup_of_different_size,
    test_bdv_pair_is_complete_or_wholly_absent,
    test_bdv_h5_without_xml_is_omitted,
    test_real_bdv_xml_full_resolution_is_unchanged_but_repointed,
    test_real_bdv_xml_crop_lands_at_the_right_micrometre,
    test_real_bdv_xml_validates_against_its_h5,
    test_validate_bdv_xml_flags_missing_h5_and_wrong_size,
    test_validate_bdv_xml_flags_setup_without_h5_group,
    test_nested_links_are_repointed_at_the_corrected_output,
    test_ambiguous_basenames_are_refused_not_guessed,
    test_unresolvable_link_takes_the_header_down_with_it,
    test_link_map_lookup_orders,
    test_matrix_pyramids_no_crop,
    test_matrix_no_pyramids_no_crop,
    test_matrix_pyramids_crop,
    test_matrix_no_pyramids_crop,
    test_validate_header_dereferences_rather_than_inspecting,
    test_validate_header_catches_size_lie_and_s1_attrs,
    test_validate_bdv_h5_catches_advertised_but_absent_level,
    test_validate_bdv_h5_catches_factor_shape_mismatch,
    test_quarantine_header_makes_it_unopenable,
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
