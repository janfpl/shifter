"""ROI extraction, shift application, and preview generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from chromatic_shift_corrector.utils import apply_integer_shift

if TYPE_CHECKING:
    import dask.array as da
    from chromatic_shift_corrector.shift_manager import ChannelTransform


def extract_subvolume(
    dask_arr: da.Array,
    z_start: int,
    z_end: int,
    y_start: int,
    y_end: int,
    x_start: int,
    x_end: int,
) -> np.ndarray:
    """Extract a sub-volume from a dask array and compute it into RAM.

    Parameters
    ----------
    dask_arr : dask.array.Array
        Full-volume dask array of shape (Z, Y, X).
    z_start, z_end : int
        Z-plane range (inclusive start, exclusive end).
    y_start, y_end : int
        Y pixel range.
    x_start, x_end : int
        X pixel range.

    Returns
    -------
    np.ndarray
        Computed numpy array of shape (z_end-z_start, y_end-y_start, x_end-x_start).
    """
    sub = dask_arr[z_start:z_end, y_start:y_end, x_start:x_end]
    return np.asarray(sub)


def generate_preview(
    dask_arr: da.Array,
    transform: ChannelTransform,
    z_start: int,
    z_end: int,
    y_start: int,
    y_end: int,
    x_start: int,
    x_end: int,
) -> np.ndarray:
    """Extract a sub-volume and apply the channel's shift.

    The shift is applied within the ROI coordinate system: shifted data that
    falls outside the ROI boundaries is lost, and the vacated region is
    zero-filled.

    Parameters
    ----------
    dask_arr : dask.array.Array
        Full-volume lazy array for this channel.
    transform : ChannelTransform
        The shift parameters for this channel.
    z_start, z_end, y_start, y_end, x_start, x_end : int
        ROI boundaries.

    Returns
    -------
    np.ndarray
        Shifted sub-volume as uint16.
    """
    sub = extract_subvolume(dask_arr, z_start, z_end, y_start, y_end, x_start, x_end)
    if transform.shift_z == 0 and transform.shift_y == 0 and transform.shift_x == 0:
        return sub
    return apply_integer_shift(sub, transform.shift_zyx)
