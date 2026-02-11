"""BigTIFF / dask loading and channel management.

The loading interface is abstracted so that future formats (e.g. HDF5) can
return dask arrays through the same API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

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


def scan_bigtiff_files(directory: Path | str) -> list[Path]:
    """Return sorted list of .tif / .tiff files in *directory*."""
    directory = Path(directory)
    files: list[Path] = []
    for ext in ("*.tif", "*.tiff"):
        files.extend(directory.glob(ext))
    return sorted(set(files))


def validate_channels(loaders: list[BigTIFFLoader]) -> tuple[bool, str]:
    """Check that all loaders have matching XY dimensions and are 3D uint16.

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


def z_dimensions_summary(loaders: list[BigTIFFLoader]) -> dict[str, int]:
    """Return a mapping of filename -> Z depth for all loaders."""
    return {loader.path.name: loader.shape[0] for loader in loaders}
