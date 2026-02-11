"""Shared utilities and metadata I/O."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from chromatic_shift_corrector import __version__

# Default colormaps in channel order.
DEFAULT_COLORMAPS = ["green", "magenta", "cyan", "yellow", "red", "blue"]

MAX_CHANNELS = 6


def apply_integer_shift(arr: np.ndarray, shift_zyx: tuple[int, int, int]) -> np.ndarray:
    """Apply an integer voxel shift to a 3D array with zero-padding (no wrap).

    Parameters
    ----------
    arr : np.ndarray
        3D array of shape (Z, Y, X).
    shift_zyx : tuple[int, int, int]
        Shift in (Z, Y, X) voxels.  Positive Z = toward higher index,
        positive Y = down, positive X = right.

    Returns
    -------
    np.ndarray
        Shifted array with same shape and dtype, zero-padded at vacated edges.
    """
    sz, sy, sx = shift_zyx
    result = np.zeros_like(arr)
    nz, ny, nx = arr.shape

    # Compute source and destination slices for each axis.
    def _slices(shift: int, length: int) -> tuple[slice, slice]:
        if shift > 0:
            src = slice(0, max(length - shift, 0))
            dst = slice(shift, length)
        elif shift < 0:
            src = slice(-shift, length)
            dst = slice(0, max(length + shift, 0))
        else:
            src = slice(0, length)
            dst = slice(0, length)
        return src, dst

    sz_src, sz_dst = _slices(sz, nz)
    sy_src, sy_dst = _slices(sy, ny)
    sx_src, sx_dst = _slices(sx, nx)

    result[sz_dst, sy_dst, sx_dst] = arr[sz_src, sy_src, sx_src]
    return result


def apply_integer_shift_2d(plane: np.ndarray, shift_yx: tuple[int, int]) -> np.ndarray:
    """Apply an integer voxel shift to a 2D array with zero-padding.

    Parameters
    ----------
    plane : np.ndarray
        2D array of shape (Y, X).
    shift_yx : tuple[int, int]
        Shift in (Y, X) voxels.

    Returns
    -------
    np.ndarray
        Shifted 2D array, zero-padded.
    """
    sy, sx = shift_yx
    result = np.zeros_like(plane)
    ny, nx = plane.shape

    def _slices(shift: int, length: int) -> tuple[slice, slice]:
        if shift > 0:
            return slice(0, max(length - shift, 0)), slice(shift, length)
        elif shift < 0:
            return slice(-shift, length), slice(0, max(length + shift, 0))
        return slice(0, length), slice(0, length)

    sy_src, sy_dst = _slices(sy, ny)
    sx_src, sx_dst = _slices(sx, nx)
    result[sy_dst, sx_dst] = plane[sy_src, sx_src]
    return result


def build_metadata(
    channels: list[dict[str, Any]],
    reference_index: int,
    voxel_xy: float,
    voxel_z: float,
    volume_shape_zyx: tuple[int, int, int],
    ram_percent: int,
) -> dict[str, Any]:
    """Build the correction metadata dictionary.

    Parameters
    ----------
    channels : list[dict]
        Each dict has keys: filename_original, filename_corrected,
        channel_index, shift_x, shift_y, shift_z.
    reference_index : int
        Index of the reference channel.
    voxel_xy : float
        XY pixel size in micrometers.
    voxel_z : float
        Z step size in micrometers.
    volume_shape_zyx : tuple[int, int, int]
        Volume dimensions as (Z, Y, X).
    ram_percent : int
        RAM allocation percentage used during export.

    Returns
    -------
    dict
        Metadata dictionary ready for JSON serialization.
    """
    ref_filename = ""
    channel_entries = []
    for ch in channels:
        is_ref = ch["channel_index"] == reference_index
        if is_ref:
            ref_filename = ch["filename_original"]
        channel_entries.append(
            {
                "filename_original": ch["filename_original"],
                "filename_corrected": ch["filename_corrected"],
                "channel_index": ch["channel_index"],
                "is_reference": is_ref,
                "shift_x_voxels": ch["shift_x"],
                "shift_y_voxels": ch["shift_y"],
                "shift_z_voxels": ch["shift_z"],
            }
        )

    nz, ny, nx = volume_shape_zyx
    return {
        "reference_channel": ref_filename,
        "voxel_size_xy_um": voxel_xy,
        "voxel_size_z_um": voxel_z,
        "volume_dimensions_xyz": [nx, ny, nz],
        "channels": channel_entries,
        "processing_date": datetime.now().isoformat(timespec="seconds"),
        "software_version": __version__,
        "ram_allocation_percent": ram_percent,
    }


def save_metadata(metadata: dict[str, Any], output_dir: Path) -> Path:
    """Write metadata dict to correction_metadata.json in *output_dir*."""
    path = output_dir / "correction_metadata.json"
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    return path


def load_metadata(path: Path) -> dict[str, Any]:
    """Load correction metadata from a JSON file."""
    with open(path) as f:
        return json.load(f)
