"""Luxendo HDF5 (.lux.h5) utilities.

Provides file management, metadata parsing, pyramid detection, and
pyramid generation for Luxendo flat-structure HDF5 volumes.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Regex for pyramid dataset names: Data_W_H_D (all integers).
_PYRAMID_RE = re.compile(r"^Data_(\d+)_(\d+)_(\d+)$")

# Sidecar header files sometimes shipped alongside Luxendo per-channel
# .lux.h5 data: an Imaris (.ims) header and a BigDataViewer HDF5/XML pair.
# Both the .ims file and the *_bdv.h5 file reference the per-channel data
# files by their literal filenames (HDF5 external links / relative XML
# paths), so they only continue to work if the exported channel files keep
# their original names in the destination directory.
_HEADER_FILE_GLOBS = ("*.ims", "*_bdv.h5", "*_bdv.xml")


def find_companion_header_files(input_dir: Path | str) -> list[Path]:
    """Return sidecar Imaris/BigDataViewer header files next to Luxendo data.

    Looks in *input_dir* for ``.ims``, ``*_bdv.h5``, and ``*_bdv.xml`` files.
    Returns an empty list if none are present (not every acquisition ships
    these).
    """
    input_dir = Path(input_dir)
    found: list[Path] = []
    for pattern in _HEADER_FILE_GLOBS:
        found.extend(sorted(input_dir.glob(pattern)))
    return found


def copy_companion_header_files(
    header_files: list[Path], output_dir: Path | str
) -> list[Path]:
    """Copy *header_files* verbatim into *output_dir*, preserving filenames.

    These headers reference the per-channel data files by their exact
    original filenames, so the corresponding exported channel files must
    also keep their original names (no ``_corrected`` suffix) for the
    copied headers to resolve correctly.
    """
    output_dir = Path(output_dir)
    copied: list[Path] = []
    for src in header_files:
        dst = output_dir / src.name
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


# --------------------------------------------------------------------------- #
# Single-resolution companion headers
#
# The Imaris (.ims) and BigDataViewer (*_bdv.h5) headers describe a
# *multi-resolution* dataset: each declares N resolution levels whose data are
# HDF5 external links into the per-channel .lux.h5 files (``Data``,
# ``Data_2_2_2``, ...). When an export writes only the full-resolution ``Data``
# (pyramids disabled), those higher-level links dangle and Imaris /
# BigDataViewer read the dataset as corrupt. The functions below rewrite a
# header to keep only the full-resolution level (level 0 -> ``Data``) so it
# resolves cleanly against pyramid-less output — e.g. for import into the Imaris
# File Converter, which builds its own pyramids from the full-resolution data.
# --------------------------------------------------------------------------- #


def _reduce_ims_to_single_resolution(ims_path: Path | str) -> bool:
    """In place, drop all but ``ResolutionLevel 0`` from an Imaris .ims header.

    Returns True if the file was recognised as an Imaris header and processed.
    """
    import h5py

    with h5py.File(str(ims_path), "r+") as f:
        if "DataSet" not in f:
            return False
        ds = f["DataSet"]
        for name in list(ds.keys()):
            if name.startswith("ResolutionLevel") and name != "ResolutionLevel 0":
                del ds[name]
    return True


def _reduce_bdv_h5_to_single_resolution(bdv_path: Path | str) -> bool:
    """In place, reduce a BigDataViewer *_bdv.h5 header to a single level.

    Truncates each setup's ``resolutions``/``subdivisions`` to the first
    (full-resolution) row and deletes every per-timepoint level group except
    ``0``. Returns True if the file looked like a BDV header and was processed.
    """
    import h5py

    with h5py.File(str(bdv_path), "r+") as f:
        setups = [k for k in f.keys() if re.fullmatch(r"s\d+", k)]
        if not setups:
            return False
        for s in setups:
            for arr in ("resolutions", "subdivisions"):
                key = f"{s}/{arr}"
                if key in f:
                    row0 = f[key][:1]
                    del f[key]
                    f.create_dataset(key, data=row0)
        for t in [k for k in f.keys() if re.fullmatch(r"t\d+", k)]:
            for s in list(f[t].keys()):
                grp = f[f"{t}/{s}"]
                for lvl in list(grp.keys()):
                    if lvl != "0":
                        del grp[lvl]
    return True


def reduce_header_to_single_resolution(path: Path | str) -> bool:
    """Reduce a single companion header (.ims / *_bdv.h5) in place to full-res.

    A *_bdv.xml is left unchanged (its resolution info lives in the paired
    *_bdv.h5). Returns True if the file was recognised and reduced.
    """
    name = Path(path).name.lower()
    try:
        if name.endswith(".ims"):
            return _reduce_ims_to_single_resolution(path)
        if name.endswith("_bdv.h5"):
            return _reduce_bdv_h5_to_single_resolution(path)
    except Exception as exc:
        logger.warning("Could not reduce %s to single resolution: %s", Path(path).name, exc)
    return False


def write_single_resolution_headers(
    header_files: list[Path],
    output_dir: Path | str,
    output_shape_zyx: tuple[int, int, int] | None = None,
) -> list[Path]:
    """Write companion headers into *output_dir* describing full-resolution only.

    Use instead of :func:`copy_companion_header_files` when the export omits
    pyramid levels. The .ims and *_bdv.h5 are **rebuilt from scratch** on the
    manufacturer's structure (see :func:`rebuild_ims_header`) rather than reduced
    in place, so the result is compact and carries no Imaris Viewer residue; a
    *_bdv.xml is copied verbatim (its resolution info lives in the .h5).
    Returns the written paths.
    """
    output_dir = Path(output_dir)
    written: list[Path] = []
    for src in header_files:
        dst = output_dir / src.name
        name = src.name.lower()
        try:
            if name.endswith(".ims"):
                rebuild_ims_header(
                    src, dst, output_shape_zyx=output_shape_zyx, keep_paths={"Data"}
                )
            elif name.endswith("_bdv.h5"):
                rebuild_bdv_h5_header(src, dst, keep_paths={"Data"})
            else:
                # *_bdv.xml carries no resolution table; copy verbatim.
                shutil.copy2(src, dst)
            written.append(dst)
        except Exception as exc:
            logger.warning(
                "Could not write single-resolution header %s: %s", src.name, exc
            )
    return written


# --------------------------------------------------------------------------- #
# ROI companion headers
#
# An ROI export writes cropped, ``_corrected_roi``-suffixed .lux.h5 files, so the
# original headers can't be used as-is: their external links point at the
# original (full-volume) filenames and their metadata carries the full-volume
# dimensions/extent. :func:`write_roi_headers` rewrites a copy of each header to
# (a) point at the ROI output files, (b) carry the ROI's voxel dimensions and a
# cropped physical extent (preserving the original voxel size), and (c) keep only
# the full-resolution level when pyramids are disabled.
# --------------------------------------------------------------------------- #


def _ims_char_attr(text: object) -> np.ndarray:
    """Encode *text* as Imaris' character-array attribute format.

    The manufacturer's headers store every text-like value (root markers,
    dimensions, extents, channel names) as a 1-D ``uint8`` array — not as an
    ``S1`` array. Both are legal HDF5, but they are different datatypes and
    Imaris compatibility takes precedence, so we emit ``uint8``.
    """
    return np.frombuffer(str(text).encode("ascii"), dtype=np.uint8).copy()


def _ims_read_char_attr(arr: object) -> str:
    """Decode an Imaris |S1 char-array attribute back to a string."""
    if isinstance(arr, bytes):
        return arr.decode("ascii", "replace")
    if isinstance(arr, str):
        return arr
    try:
        return arr.tobytes().decode("ascii", "replace")  # type: ignore[attr-defined]
    except Exception:
        return str(arr)


def _set_ims_char_attr(grp: Any, key: str, text: object) -> None:
    if key in grp.attrs:
        del grp.attrs[key]
    grp.attrs.create(key, _ims_char_attr(text))


def _remap_ims_external_links(f: Any, rename: Any) -> None:
    """Rewrite each Channel's ``Data`` external-link filename via *rename*."""
    import h5py

    ds = f.get("DataSet")
    if ds is None:
        return
    for lvl_name in ds.keys():
        lvl = ds[lvl_name]
        for tp_name in lvl.keys():
            tp = lvl[tp_name]
            for ch_name in tp.keys():
                grp = tp[ch_name]
                link = grp.get("Data", getlink=True)
                if isinstance(link, h5py.ExternalLink):
                    new = rename(link.filename)
                    if new != link.filename:
                        path = link.path
                        del grp["Data"]
                        grp["Data"] = h5py.ExternalLink(new, path)


def _remap_bdv_external_links(f: Any, rename: Any) -> None:
    """Rewrite each ``cells`` external-link filename via *rename*."""
    import h5py

    for t in [k for k in f.keys() if re.fullmatch(r"t\d+", k)]:
        for s in f[t].keys():
            grp = f[f"{t}/{s}"]
            for lvl in grp.keys():
                cell_grp = grp[lvl]
                link = cell_grp.get("cells", getlink=True)
                if isinstance(link, h5py.ExternalLink):
                    new = rename(link.filename)
                    if new != link.filename:
                        path = link.path
                        del cell_grp["cells"]
                        cell_grp["cells"] = h5py.ExternalLink(new, path)


def _crop_ims_dimensions(
    f: Any, roi: tuple[int, int, int, int, int, int]
) -> None:
    """Set an Imaris header's dimensions/extent to the ROI, preserving voxel size.

    Imaris stores voxel dimensions in ``DataSetInfo/Image`` as X/Y/Z and the
    physical bounding box as ExtMin{0,1,2}/ExtMax{0,1,2} (axis 0=X, 1=Y, 2=Z),
    where voxel size = (ExtMax-ExtMin)/N. The ROI keeps the same voxel size, so
    the new extent is the corresponding sub-box.
    """
    z0, z1, y0, y1, x0, x1 = roi
    img = f.get("DataSetInfo/Image")
    if img is None:
        return

    def _read(k: str) -> float:
        return float(_ims_read_char_attr(img.attrs[k]))

    axes = {0: ("X", x0, x1), 1: ("Y", y0, y1), 2: ("Z", z0, z1)}
    for i, (dim_name, start, end) in axes.items():
        n = _read(dim_name)
        mn = _read(f"ExtMin{i}")
        mx = _read(f"ExtMax{i}")
        vox = (mx - mn) / n if n else 0.0
        _set_ims_char_attr(img, f"ExtMin{i}", f"{mn + start * vox:.6g}")
        _set_ims_char_attr(img, f"ExtMax{i}", f"{mn + end * vox:.6g}")
    _set_ims_char_attr(img, "X", x1 - x0)
    _set_ims_char_attr(img, "Y", y1 - y0)
    _set_ims_char_attr(img, "Z", z1 - z0)


def write_roi_headers(
    header_files: list[Path],
    output_dir: Path | str,
    output_suffix: str,
    roi: tuple[int, int, int, int, int, int],
    write_pyramids: bool,
) -> list[Path]:
    """Write companion headers matching an ROI-cropped export.

    Each ``.ims`` / ``*_bdv.h5`` is copied, its external links repointed to the
    ``output_suffix``-named ROI output files, reduced to a single resolution when
    *write_pyramids* is False, and (for ``.ims``) given the ROI's dimensions and
    a cropped extent. A ``*_bdv.xml`` is skipped: its ViewSetup dimensions can't
    be regenerated reliably here, and Imaris (the primary consumer) uses the
    ``.ims``. Returns the written paths.
    """
    from shifter.utils import h5_output_filename

    output_dir = Path(output_dir)

    def rename(filename: str) -> str:
        return h5_output_filename(filename, output_suffix)

    z0, z1, y0, y1, x0, x1 = roi
    out_shape = (z1 - z0, y1 - y0, x1 - x0)
    keep_paths = None if write_pyramids else {"Data"}

    written: list[Path] = []
    for src in header_files:
        name = src.name.lower()
        dst = output_dir / src.name
        try:
            if name.endswith(".ims"):
                # Rebuilding (rather than editing a copy) updates *every*
                # dimension declaration: DataSetInfo/Image X/Y/Z, the cropped
                # physical extent, and each level's own ImageSizeX/Y/Z derived
                # from that level's downsample factors — so no part of the
                # header contradicts the data it links to.
                rebuild_ims_header(
                    src, dst, rename=rename, output_shape_zyx=out_shape,
                    roi=roi, keep_paths=keep_paths,
                )
                written.append(dst)
            elif name.endswith("_bdv.h5"):
                rebuild_bdv_h5_header(src, dst, rename=rename, keep_paths=keep_paths)
                written.append(dst)
            else:
                logger.info(
                    "Skipping %s for ROI export (BDV XML dimensions not regenerated).",
                    src.name,
                )
        except Exception as exc:
            logger.warning("Could not write ROI header %s: %s", src.name, exc)
    return written


class H5FileManager:
    """Manages open h5py file handles for dask-backed lazy loading.

    h5py file handles **must** stay open while dask arrays reference them.
    This class keeps a registry of open files and closes them all on demand.
    """

    def __init__(self) -> None:
        self._open_files: dict[str, Any] = {}  # str(path) → h5py.File

    def open(self, filepath: Path | str) -> Any:
        """Open (or reuse) an h5py File handle in read mode."""
        import h5py

        key = str(filepath)
        if key not in self._open_files:
            self._open_files[key] = h5py.File(key, "r")
        return self._open_files[key]

    def close_all(self) -> None:
        """Close all open file handles."""
        for f in self._open_files.values():
            try:
                f.close()
            except Exception:
                pass
        self._open_files.clear()


def scan_h5_files(directory: Path | str) -> list[Path]:
    """Return sorted list of .h5/.lux.h5 files, ignoring 'main*' files.

    Files whose name starts with "main" (case-insensitive) are header
    files and should be ignored.  JSON sidecars are also skipped.
    """
    directory = Path(directory)
    files: list[Path] = []
    for p in directory.iterdir():
        if not p.is_file():
            continue
        name_lower = p.name.lower()
        # Skip non-H5
        if not (name_lower.endswith(".h5") or name_lower.endswith(".lux.h5")):
            continue
        # Skip main header files
        if name_lower.startswith("main"):
            continue
        files.append(p)
    return sorted(files)


def parse_h5_metadata(h5file: Any) -> dict[str, Any]:
    """Parse the ``metadata`` dataset from a Luxendo H5 file.

    Returns a dict with extracted fields, or an empty dict if metadata is
    missing or unparseable.  Keys that may be present:

    - ``voxel_size_xy_um`` : float
    - ``voxel_size_z_um`` : float
    - ``image_size_vx`` : dict with width/height/depth
    - ``channel_description`` : str
    - ``channel_id`` : str/int
    - ``raw_metadata`` : the full parsed JSON dict
    """
    result: dict[str, Any] = {}

    if "metadata" not in h5file:
        return result

    try:
        raw = h5file["metadata"][()]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        meta = json.loads(raw)
        result["raw_metadata"] = meta
    except Exception as exc:
        logger.warning("Failed to parse H5 metadata: %s", exc)
        return result

    proc = meta.get("processingInformation", {})

    # Voxel sizes.
    voxel = proc.get("voxel_size_um", {})
    if isinstance(voxel, dict):
        # width = X, height = Y, depth = Z.
        # XY pixel size: use width (X dimension).
        try:
            result["voxel_size_xy_um"] = float(voxel.get("width", 0))
            result["voxel_size_z_um"] = float(voxel.get("depth", 0))
        except (TypeError, ValueError):
            pass

    # Image size.
    img_size = proc.get("image_size_vx", {})
    if isinstance(img_size, dict):
        result["image_size_vx"] = img_size

    # Channel description.
    ch_desc = proc.get("channel_description", "")
    if ch_desc:
        result["channel_description"] = str(ch_desc)

    ch_id = proc.get("channel_id", "")
    if ch_id:
        result["channel_id"] = ch_id

    return result


def detect_pyramid_levels(h5file: Any) -> list[tuple[str, int, int, int]]:
    """Detect pyramid datasets in a Luxendo H5 file.

    Returns a list of ``(dataset_name, factor_w, factor_h, factor_d)`` tuples
    sorted by total downsample factor ascending.  ``Data`` (full-res) is NOT
    included.  Factor order: W=X, H=Y, D=Z.
    """
    levels: list[tuple[str, int, int, int]] = []
    for name in h5file.keys():
        m = _PYRAMID_RE.match(name)
        if m:
            fw, fh, fd = int(m.group(1)), int(m.group(2)), int(m.group(3))
            levels.append((name, fw, fh, fd))

    # Sort by total factor (product of all three).
    levels.sort(key=lambda t: t[1] * t[2] * t[3])
    return levels


def block_average_3d(
    slab: np.ndarray,
    factor_w: int,
    factor_h: int,
    factor_d: int,
) -> np.ndarray:
    """Downsample a 3D slab via block averaging.

    Parameters
    ----------
    slab : np.ndarray
        Input array of shape ``(D_planes, H, W)`` where D_planes == factor_d.
    factor_w : int
        Downsample factor for width (X axis).
    factor_h : int
        Downsample factor for height (Y axis).
    factor_d : int
        Downsample factor for depth (Z axis).

    Returns
    -------
    np.ndarray
        Downsampled 2D plane of shape ``(H // factor_h, W // factor_w)``
        as uint16.
    """
    d, h, w = slab.shape
    h_trim = (h // factor_h) * factor_h
    w_trim = (w // factor_w) * factor_w
    trimmed = slab[:factor_d, :h_trim, :w_trim].astype(np.float64)
    reshaped = trimmed.reshape(
        factor_d, h_trim // factor_h, factor_h, w_trim // factor_w, factor_w
    )
    return reshaped.mean(axis=(0, 2, 4)).astype(np.uint16)


def compute_pyramid_level_shape(
    data_shape: tuple[int, int, int], factor_w: int, factor_h: int, factor_d: int
) -> tuple[int, int, int]:
    """Return the (nz, ny, nx) shape :func:`generate_pyramid_level` will produce."""
    nz, ny, nx = data_shape
    return nz // factor_d, ny // factor_h, nx // factor_w


def generate_pyramid_level(
    corrected_h5: Any,
    level_name: str,
    factor_w: int,
    factor_h: int,
    factor_d: int,
    chunks: tuple[int, ...] | None = None,
) -> None:
    """Generate a single pyramid level from the corrected ``Data`` dataset.

    Reads factor_d consecutive Z-planes at a time from ``Data``, averages
    them into one output Z-plane (also averaging XY), and writes to
    *level_name* dataset.  Memory-efficient: only factor_d input planes
    are in memory at a time.

    Parameters
    ----------
    corrected_h5 : h5py.File
        Open HDF5 file (writable) containing the corrected ``Data`` dataset.
    level_name : str
        Output dataset name (e.g. ``"Data_2_2_2"``).
    factor_w, factor_h, factor_d : int
        Downsample factors for X, Y, Z respectively.
    chunks : tuple, optional
        Chunk shape for the output dataset.  If None, uses (64, 64, 64)
        clamped to the output dimensions.

    Returns
    -------
    dict or None
        A timing/throughput breakdown for the level with keys ``read_s``,
        ``compute_s``, ``write_s`` (seconds spent reading ``Data`` slabs,
        block-averaging, and writing output planes), ``bytes_read``,
        ``bytes_written``, and ``out_shape``. Returns ``None`` if the level is
        skipped (a zero output dimension). The breakdown lets callers log where
        the pyramid phase spends its time (disk reads vs. compute) so the
        bottleneck can be identified before optimising.
    """
    data = corrected_h5["Data"]
    nz, ny, nx = data.shape

    out_nz = nz // factor_d
    out_ny = ny // factor_h
    out_nx = nx // factor_w

    if out_nz == 0 or out_ny == 0 or out_nx == 0:
        logger.warning(
            "Pyramid level %s: output dimensions would be zero, skipping.", level_name
        )
        return None

    if chunks is None:
        chunks = (min(64, out_nz), min(64, out_ny), min(64, out_nx))
    else:
        # Clamp chunk dimensions so they never exceed the data shape.
        chunks = (min(chunks[0], out_nz), min(chunks[1], out_ny), min(chunks[2], out_nx))

    if level_name in corrected_h5:
        del corrected_h5[level_name]

    ds = corrected_h5.create_dataset(
        level_name,
        shape=(out_nz, out_ny, out_nx),
        dtype=np.uint16,
        chunks=chunks,
    )

    read_s = compute_s = write_s = 0.0
    for oz in range(out_nz):
        iz_start = oz * factor_d
        iz_end = iz_start + factor_d

        t0 = time.perf_counter()
        slab = data[iz_start:iz_end, :, :]
        if not isinstance(slab, np.ndarray):
            slab = np.array(slab)
        t1 = time.perf_counter()
        plane = block_average_3d(slab, factor_w, factor_h, factor_d)
        t2 = time.perf_counter()
        ds[oz, :, :] = plane
        t3 = time.perf_counter()

        read_s += t1 - t0
        compute_s += t2 - t1
        write_s += t3 - t2

    return {
        "read_s": read_s,
        "compute_s": compute_s,
        "write_s": write_s,
        # Input planes actually read for this level (whole Data minus the
        # remainder planes dropped by integer division).
        "bytes_read": out_nz * factor_d * ny * nx * 2,
        "bytes_written": out_nz * out_ny * out_nx * 2,
        "out_shape": (out_nz, out_ny, out_nx),
    }


# --------------------------------------------------------------------------- #
# Manufacturer-style header rebuilding
#
# Reducing a header by deleting resolution levels in place has two problems:
# HDF5 does not reclaim the deleted space, and — more importantly — any
# ancillary structure the source header happens to carry is preserved. Headers
# that have been opened and re-saved by Imaris Viewer gain native-Imaris markers
# (``Thumbnail``, ``VolumeMask``, ``DataSetTimes``, ``DataSetEvents``,
# ``DataSetInfo/Imaris`` ...) that the manufacturer's own headers do not have,
# which can make Imaris treat an external-link header as a native dataset.
#
# The builders below instead write a NEW header containing exactly the structure
# the manufacturer emits: minimal root markers, the requested resolution levels
# (each channel carrying its own ImageSizeX/Y/Z), and DataSetInfo with only
# Channel/Image/TimeInfo. Text attributes are written as uint8 arrays to match.
# The result is compact, and is written to a temporary file and moved into place
# so a failure never leaves a half-written header.
# --------------------------------------------------------------------------- #

_IMS_ROOT_ATTR_DEFAULTS: dict[str, str] = {
    "DataSetDirectoryName": "DataSet",
    "DataSetInfoDirectoryName": "DataSetInfo",
    "ImarisDataSet": "ImarisDataSet",
    "ImarisVersion": "5.5.0",
    "ThumbnailDirectoryName": "Thumbnail",
}

# DataSetInfo children the manufacturer emits; anything else (Imaris,
# ImarisDataSet, Log, ...) is Imaris Viewer residue and is dropped.
_IMS_INFO_KEEP_RE = re.compile(r"^(Channel \d+|Image|TimeInfo)$")


def dataset_path_factors(link_path: str) -> tuple[int, int, int]:
    """Return ``(factor_w, factor_h, factor_d)`` for a Luxendo dataset path.

    ``Data`` is full resolution -> ``(1, 1, 1)``; ``Data_W_H_D`` -> ``(W, H, D)``.
    Factors are derived from the link target rather than assumed from the
    resolution-level index, which need not be a power of two.
    """
    m = _PYRAMID_RE.match(link_path.strip("/").split("/")[-1])
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return (1, 1, 1)


def _sorted_resolution_levels(dataset_grp: Any) -> list[str]:
    """Return ``ResolutionLevel N`` names sorted by their numeric index."""
    def _idx(name: str) -> int:
        try:
            return int(name.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            return 0

    return sorted(
        (k for k in dataset_grp.keys() if k.startswith("ResolutionLevel")), key=_idx
    )


def _copy_attrs_as_ims_text(src_grp: Any, dst_grp: Any, skip: set[str] | None = None) -> None:
    """Copy *src_grp*'s attributes to *dst_grp* in Imaris uint8 text form."""
    skip = skip or set()
    for key, value in src_grp.attrs.items():
        if key in skip:
            continue
        dst_grp.attrs.create(key, _ims_char_attr(_ims_read_char_attr(value)))


def rebuild_ims_header(
    src_path: Path | str,
    dst_path: Path | str,
    *,
    rename: Any = None,
    output_shape_zyx: tuple[int, int, int] | None = None,
    roi: tuple[int, int, int, int, int, int] | None = None,
    keep_paths: set[str] | None = None,
) -> dict[str, Any]:
    """Write a fresh, manufacturer-style Imaris header derived from *src_path*.

    Parameters
    ----------
    rename : callable, optional
        Maps each external-link filename to the exported filename.
    output_shape_zyx : tuple, optional
        ``(nz, ny, nx)`` of the exported full-resolution data. Each channel's
        ``ImageSizeX/Y/Z`` is derived from this and the level's own downsample
        factors, so every declared size matches the dataset it links to.
    roi : tuple, optional
        ``(z0, z1, y0, y1, x0, x1)``; crops ``DataSetInfo/Image``'s physical
        extent while preserving voxel size.
    keep_paths : set[str], optional
        Dataset paths that actually exist in the output (e.g. ``{"Data"}`` when
        pyramids were skipped). Levels linking to anything else are omitted, so
        the header never advertises a level that was not written.

    Returns
    -------
    dict
        ``{"levels": [...], "channels": int}`` describing what was written.
    """
    import h5py

    src_path = Path(src_path)
    dst_path = Path(dst_path)
    tmp_path = dst_path.with_name(dst_path.name + ".tmp")

    written_levels: list[str] = []
    n_channels = 0

    with h5py.File(str(src_path), "r") as src, h5py.File(str(tmp_path), "w") as dst:
        # --- root markers ------------------------------------------------
        for key, default in _IMS_ROOT_ATTR_DEFAULTS.items():
            value = (
                _ims_read_char_attr(src.attrs[key]) if key in src.attrs else default
            )
            dst.attrs.create(key, _ims_char_attr(value))

        # --- DataSet ------------------------------------------------------
        src_ds = src["DataSet"]
        dst_ds = dst.create_group("DataSet")
        out_index = 0
        for level_name in _sorted_resolution_levels(src_ds):
            src_level = src_ds[level_name]

            # Decide whether this level survives, based on what it links to.
            level_paths: set[str] = set()
            for tp in src_level.keys():
                for ch in src_level[tp].keys():
                    link = src_level[tp][ch].get("Data", getlink=True)
                    if isinstance(link, h5py.ExternalLink):
                        level_paths.add(link.path.strip("/").split("/")[-1])
            if keep_paths is not None and not (level_paths & keep_paths):
                continue

            dst_level = dst_ds.create_group(f"ResolutionLevel {out_index}")
            out_index += 1
            written_levels.append(level_name)

            for tp_name in sorted(src_level.keys()):
                src_tp = src_level[tp_name]
                dst_tp = dst_level.create_group(tp_name)
                for ch_name in sorted(src_tp.keys()):
                    src_ch = src_tp[ch_name]
                    dst_ch = dst_tp.create_group(ch_name)
                    link = src_ch.get("Data", getlink=True)

                    factors = (1, 1, 1)
                    if isinstance(link, h5py.ExternalLink):
                        factors = dataset_path_factors(link.path)
                        target = link.filename
                        if rename is not None:
                            target = rename(target)
                        dst_ch["Data"] = h5py.ExternalLink(target, link.path)

                    # Histogram payload + range markers.
                    if "Histogram" in src_ch:
                        dst_ch.create_dataset(
                            "Histogram", data=src_ch["Histogram"][()]
                        )
                    for key, default in (("HistogramMin", "0"), ("HistogramMax", "65535")):
                        value = (
                            _ims_read_char_attr(src_ch.attrs[key])
                            if key in src_ch.attrs
                            else default
                        )
                        dst_ch.attrs.create(key, _ims_char_attr(value))

                    # Per-level dimensions, derived from the linked level's
                    # factors so they always agree with the linked dataset.
                    if output_shape_zyx is not None:
                        nz, ny, nx = output_shape_zyx
                        fw, fh, fd = factors
                        dst_ch.attrs.create("ImageSizeX", _ims_char_attr(nx // fw))
                        dst_ch.attrs.create("ImageSizeY", _ims_char_attr(ny // fh))
                        dst_ch.attrs.create("ImageSizeZ", _ims_char_attr(nz // fd))
                    else:
                        for key in ("ImageSizeX", "ImageSizeY", "ImageSizeZ"):
                            if key in src_ch.attrs:
                                dst_ch.attrs.create(
                                    key, _ims_char_attr(_ims_read_char_attr(src_ch.attrs[key]))
                                )
                    n_channels = max(n_channels, len(src_tp.keys()))

        # --- DataSetInfo (Channel N / Image / TimeInfo only) --------------
        dst_info = dst.create_group("DataSetInfo")
        if "DataSetInfo" in src:
            src_info = src["DataSetInfo"]
            for name in src_info.keys():
                if not _IMS_INFO_KEEP_RE.match(name):
                    continue  # drop Imaris Viewer residue
                _copy_attrs_as_ims_text(src_info[name], dst_info.create_group(name))

            # Overall dimensions / extent.
            if "Image" in dst_info and "Image" in src_info:
                dst_image = dst_info["Image"]
                src_image = src_info["Image"]

                def _num(key: str) -> float | None:
                    if key not in src_image.attrs:
                        return None
                    try:
                        return float(_ims_read_char_attr(src_image.attrs[key]))
                    except ValueError:
                        return None

                if roi is not None:
                    z0, z1, y0, y1, x0, x1 = roi
                    for axis, dim_key, start, end in (
                        (0, "X", x0, x1), (1, "Y", y0, y1), (2, "Z", z0, z1)
                    ):
                        n = _num(dim_key)
                        mn = _num(f"ExtMin{axis}")
                        mx = _num(f"ExtMax{axis}")
                        if None in (n, mn, mx) or not n:
                            continue
                        vox = (mx - mn) / n
                        dst_image.attrs.create(
                            f"ExtMin{axis}", _ims_char_attr(f"{mn + start * vox:.6g}")
                        )
                        dst_image.attrs.create(
                            f"ExtMax{axis}", _ims_char_attr(f"{mn + end * vox:.6g}")
                        )

                if output_shape_zyx is not None:
                    nz, ny, nx = output_shape_zyx
                    dst_image.attrs.create("X", _ims_char_attr(nx))
                    dst_image.attrs.create("Y", _ims_char_attr(ny))
                    dst_image.attrs.create("Z", _ims_char_attr(nz))

    os.replace(str(tmp_path), str(dst_path))
    return {"levels": written_levels, "channels": n_channels}


def rebuild_bdv_h5_header(
    src_path: Path | str,
    dst_path: Path | str,
    *,
    rename: Any = None,
    keep_paths: set[str] | None = None,
) -> dict[str, Any]:
    """Write a fresh BigDataViewer HDF5 header derived from *src_path*.

    Keeps only the levels whose ``cells`` link targets are in *keep_paths* (all
    levels when it is None), truncating each setup's ``resolutions`` /
    ``subdivisions`` tables to match, and remapping link filenames via *rename*.
    """
    import h5py

    src_path = Path(src_path)
    dst_path = Path(dst_path)
    tmp_path = dst_path.with_name(dst_path.name + ".tmp")

    kept_per_setup: dict[str, list[str]] = {}

    with h5py.File(str(src_path), "r") as src, h5py.File(str(tmp_path), "w") as dst:
        timepoints = [k for k in src.keys() if re.fullmatch(r"t\d+", k)]

        for t in sorted(timepoints):
            for s in sorted(src[t].keys()):
                src_grp = src[f"{t}/{s}"]
                keep: list[str] = []
                for lvl in sorted(src_grp.keys(), key=lambda v: int(v) if v.isdigit() else 0):
                    link = src_grp[lvl].get("cells", getlink=True)
                    if not isinstance(link, h5py.ExternalLink):
                        continue
                    leaf = link.path.strip("/").split("/")[-1]
                    if keep_paths is not None and leaf not in keep_paths:
                        continue
                    target = link.filename
                    if rename is not None:
                        target = rename(target)
                    dst.create_group(f"{t}/{s}/{len(keep)}")
                    dst[f"{t}/{s}/{len(keep)}/cells"] = h5py.ExternalLink(
                        target, link.path
                    )
                    keep.append(lvl)
                kept_per_setup.setdefault(s, keep)

        # Resolution tables truncated to the number of surviving levels.
        for s in [k for k in src.keys() if re.fullmatch(r"s\d+", k)]:
            n_keep = len(kept_per_setup.get(s, []))
            for arr in ("resolutions", "subdivisions"):
                key = f"{s}/{arr}"
                if key not in src:
                    continue
                data = src[key][:]
                if n_keep:
                    data = data[:n_keep]
                dst.create_dataset(key, data=data)

    os.replace(str(tmp_path), str(dst_path))
    return {"levels_per_setup": kept_per_setup}
