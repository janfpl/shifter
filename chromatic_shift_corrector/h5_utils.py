"""Luxendo HDF5 (.lux.h5) utilities.

Provides file management, metadata parsing, pyramid detection, and
pyramid generation for Luxendo flat-structure HDF5 volumes.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
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
        return

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

    for oz in range(out_nz):
        iz_start = oz * factor_d
        iz_end = iz_start + factor_d
        slab = data[iz_start:iz_end, :, :]
        if not isinstance(slab, np.ndarray):
            slab = np.array(slab)
        ds[oz, :, :] = block_average_3d(slab, factor_w, factor_h, factor_d)
