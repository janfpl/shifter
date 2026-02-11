"""Preprocessing steps applied to sub-volumes before registration.

These operate on **copies** and never modify the original data.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

# Hardcoded defaults (v1 — not user-adjustable).
_BG_PERCENTILE = 5
_GAUSSIAN_SIGMA = (2.0, 2.0, 1.0)  # (Z, Y, X) — anisotropic


def subtract_background(volume: np.ndarray) -> np.ndarray:
    """Percentile-based background subtraction.

    Computes the 5th-percentile intensity and subtracts it, clipping at zero.
    """
    bg = np.percentile(volume, _BG_PERCENTILE)
    out = volume.astype(np.float64) - bg
    np.clip(out, 0, None, out=out)
    return out.astype(volume.dtype)


def smooth_gaussian(volume: np.ndarray) -> np.ndarray:
    """3-D Gaussian smoothing with anisotropic sigma (2, 2, 1)."""
    return gaussian_filter(volume.astype(np.float64), sigma=_GAUSSIAN_SIGMA).astype(
        volume.dtype
    )


def preprocess(
    volume: np.ndarray,
    background_subtraction: bool = False,
    gaussian_smoothing: bool = False,
    use_gpu: bool = False,
) -> np.ndarray:
    """Apply preprocessing pipeline to a copy of *volume*.

    Order: background subtraction first, then Gaussian smoothing.

    Parameters
    ----------
    volume : np.ndarray
        3-D array (Z, Y, X).
    background_subtraction : bool
        Enable percentile-based background subtraction.
    gaussian_smoothing : bool
        Enable 3-D Gaussian smoothing.
    use_gpu : bool
        If True and cupy is available, use GPU-accelerated versions.

    Returns
    -------
    np.ndarray
        Preprocessed copy (always on CPU).
    """
    vol = volume.copy()

    if use_gpu:
        try:
            return _preprocess_gpu(vol, background_subtraction, gaussian_smoothing)
        except Exception:
            pass  # fall through to CPU

    if background_subtraction:
        vol = subtract_background(vol)
    if gaussian_smoothing:
        vol = smooth_gaussian(vol)
    return vol


def _preprocess_gpu(
    volume: np.ndarray,
    background_subtraction: bool,
    gaussian_smoothing: bool,
) -> np.ndarray:
    """GPU path — uses cupy for both operations."""
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter as gpu_gaussian_filter

    vol = cp.asarray(volume)

    if background_subtraction:
        bg = float(cp.percentile(vol, _BG_PERCENTILE))
        vol = vol.astype(cp.float64) - bg
        cp.clip(vol, 0, None, out=vol)
        vol = vol.astype(volume.dtype)

    if gaussian_smoothing:
        vol = gpu_gaussian_filter(
            vol.astype(cp.float64), sigma=_GAUSSIAN_SIGMA
        ).astype(volume.dtype)

    return vol.get()  # back to CPU
