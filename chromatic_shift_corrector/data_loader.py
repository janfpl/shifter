"""BigTIFF / Luxendo H5 / dask loading and channel management.

The loading interface is abstracted so that different formats return dask
arrays through the same API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import dask.array as da
import numpy as np
import tifffile


class VolumeLoader(Protocol):
    """Protocol for lazy volume loaders.

    Any loader must expose a dask array through :pyattr:`dask_array` and
    provide basic shape / dtype information.
    """

    @property
    def dask_array(self) -> da.Array: ...

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> np.dtype: ...

    def close(self) -> None: ...


class BigTIFFLoader:
    """Lazy loader for a single-channel 3D BigTIFF volume.

    The file is memory-mapped via :mod:`tifffile` and wrapped in a dask array
    so that only the currently requested slices are read from disk.

    Parameters
    ----------
    path : Path | str
        Path to a BigTIFF file.
    chunk_z : int
        Number of Z-planes per dask chunk (default 64).
    """

    def __init__(self, path: Path | str, chunk_z: int = 64) -> None:
        self.path = Path(path)
        self._store = tifffile.imread(str(self.path), aszarr=True)
        self._zarr = da.from_zarr(self._store)
        # Ensure 3D: tifffile may return (Z, Y, X) or (Y, X) for single-plane
        if self._zarr.ndim == 2:
            self._zarr = self._zarr[np.newaxis, ...]
        # Re-chunk along Z for efficient slab reads
        self._zarr = self._zarr.rechunk({0: chunk_z, 1: -1, 2: -1})

    @property
    def dask_array(self) -> da.Array:
        return self._zarr

    @property
    def shape(self) -> tuple[int, ...]:
        return self._zarr.shape

    @property
    def dtype(self) -> np.dtype:
        return self._zarr.dtype

    def close(self) -> None:
        if hasattr(self._store, "close"):
            self._store.close()


class H5Loader:
    """Lazy loader for a single-channel Luxendo .lux.h5 volume.

    Uses :class:`h5_utils.H5FileManager` to keep h5py file handles open
    so that dask arrays can access data lazily.  Also provides multiscale
    dask arrays for napari pyramid rendering.

    Parameters
    ----------
    path : Path | str
        Path to a .lux.h5 file.
    file_manager : h5_utils.H5FileManager
        Shared file-handle manager.
    chunk_z : int
        Number of Z-planes per dask chunk (default 64).
    """

    def __init__(
        self,
        path: Path | str,
        file_manager: Any,
        chunk_z: int = 64,
    ) -> None:
        from chromatic_shift_corrector.h5_utils import (
            detect_pyramid_levels,
            parse_h5_metadata,
        )

        self.path = Path(path)
        self._file_manager = file_manager
        self._h5 = file_manager.open(self.path)

        if "Data" not in self._h5:
            raise ValueError(
                f"{self.path.name}: not a flat Luxendo H5 file "
                "(missing 'Data' dataset)."
            )

        # Full-resolution dask array.
        data_ds = self._h5["Data"]
        self._dask = da.from_array(data_ds, chunks=(chunk_z, -1, -1))

        # Ensure 3D.
        if self._dask.ndim == 2:
            self._dask = self._dask[np.newaxis, ...]

        # Multiscale arrays (full-res + pyramid levels).
        self._multiscale: list[da.Array] = [self._dask]
        self._pyramid_levels = detect_pyramid_levels(self._h5)
        for level_name, _fw, _fh, _fd in self._pyramid_levels:
            ds = self._h5[level_name]
            arr = da.from_array(ds, chunks=(min(chunk_z, ds.shape[0]), -1, -1))
            self._multiscale.append(arr)

        # Parse metadata.
        self._metadata = parse_h5_metadata(self._h5)

    @property
    def dask_array(self) -> da.Array:
        """Full-resolution dask array (Z, Y, X)."""
        return self._dask

    @property
    def multiscale(self) -> list[da.Array]:
        """List of dask arrays [full_res, 2x, 3x, ...] for napari."""
        return self._multiscale

    @property
    def shape(self) -> tuple[int, ...]:
        return self._dask.shape

    @property
    def dtype(self) -> np.dtype:
        return self._dask.dtype

    @property
    def h5_metadata(self) -> dict[str, Any]:
        """Parsed Luxendo metadata dict."""
        return self._metadata

    @property
    def pyramid_levels(self) -> list[tuple[str, int, int, int]]:
        """List of (name, factor_w, factor_h, factor_d) pyramid levels."""
        return self._pyramid_levels

    @property
    def num_levels(self) -> int:
        """Total number of resolution levels (including full-res)."""
        return len(self._multiscale)

    def multiscale_subset(
        self, min_level: int = 0, max_level: int | None = None
    ) -> list[da.Array]:
        """Return a slice of the multiscale list from *min_level* to *max_level* (inclusive).

        Level 0 is full resolution; higher indices are progressively downsampled.
        """
        if max_level is None:
            max_level = len(self._multiscale) - 1
        return self._multiscale[min_level : max_level + 1]

    def level_descriptions(self) -> list[str]:
        """Return human-readable descriptions for each pyramid level."""
        descs = ["Level 0: Full resolution"]
        for i, (name, fw, fh, fd) in enumerate(self._pyramid_levels):
            descs.append(f"Level {i + 1}: {fw}x{fh}x{fd} downsample")
        return descs

    @property
    def channel_description(self) -> str:
        """Channel description from metadata, or filename as fallback."""
        return self._metadata.get("channel_description", self.path.name)

    def close(self) -> None:
        # File handles are managed by H5FileManager, not closed individually.
        pass


def scan_bigtiff_files(directory: Path | str) -> list[Path]:
    """Return sorted list of .tif / .tiff files in *directory*."""
    directory = Path(directory)
    files: list[Path] = []
    for ext in ("*.tif", "*.tiff"):
        files.extend(directory.glob(ext))
    return sorted(set(files))


def validate_channels(loaders: list[Any]) -> tuple[bool, str]:
    """Check that all loaders have matching XY dimensions and are 3D uint16.

    Works with both ``BigTIFFLoader`` and ``H5Loader``.

    Returns
    -------
    (ok, message) : tuple[bool, str]
        ``ok`` is True if validation passes; *message* describes the issue
        otherwise.
    """
    if not loaders:
        return False, "No channels loaded."

    ref_shape = loaders[0].shape
    ref_dtype = loaders[0].dtype

    if ref_dtype != np.uint16:
        return False, f"Expected uint16 data, got {ref_dtype}."

    for i, loader in enumerate(loaders[1:], start=1):
        if loader.dtype != np.uint16:
            return False, f"Channel {i} has dtype {loader.dtype}, expected uint16."
        if loader.shape[1:] != ref_shape[1:]:
            return (
                False,
                f"Channel {i} XY dimensions {loader.shape[1:]} differ from "
                f"channel 0 dimensions {ref_shape[1:]}.",
            )

    return True, "OK"


def z_dimensions_summary(loaders: list[Any]) -> dict[str, int]:
    """Return a mapping of filename -> Z depth for all loaders."""
    return {loader.path.name: loader.shape[0] for loader in loaders}
