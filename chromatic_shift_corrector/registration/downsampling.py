"""Placeholder for future full-volume downsampled registration.

The registration runner accepts a numpy array (the sub-volume to register).
Currently this array always comes from ROI extraction.  In a future version,
this module will provide a function that produces a downsampled array from
the full volume so registration can run on the whole dataset without
exhausting RAM.
"""

from __future__ import annotations

import numpy as np


def downsample_full_volume(
    dask_array,
    factor_xy: int = 4,
    factor_z: int = 2,
) -> np.ndarray:
    """Downsample a full-volume dask array for registration.

    .. note::
        This is a **stub** for future implementation. Calling it will raise
        ``NotImplementedError``.

    Parameters
    ----------
    dask_array : dask.array.Array
        Full-volume lazy array of shape (Z, Y, X).
    factor_xy : int
        Downsampling factor in X and Y (default 4).
    factor_z : int
        Downsampling factor in Z (default 2).

    Returns
    -------
    np.ndarray
        Downsampled 3-D numpy array suitable for passing to a registration
        algorithm.

    Raises
    ------
    NotImplementedError
        Always — this is a placeholder for future work.
    """
    raise NotImplementedError(
        "Full-volume downsampled registration is not yet implemented. "
        "Use ROI-based sub-volume registration instead."
    )
