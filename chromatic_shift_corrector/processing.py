"""Image processing algorithms: rolling ball, gaussian blur, unsharp mask.

Each filter operates 2D per-slice (matching ImageJ behavior). The fallback
chain per filter is: cupy GPU -> scipy CPU.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import scipy.ndimage

logger = logging.getLogger(__name__)


def _get_cupy_ndimage() -> tuple[Any, Any] | None:
    """Return (cupy, cupyx.scipy.ndimage) or None if unavailable."""
    try:
        import cupy
        import cupyx.scipy.ndimage as cupy_ndi
        return cupy, cupy_ndi
    except ImportError:
        return None


def _make_disk_footprint(radius: int) -> np.ndarray:
    """Create a 2D disk structuring element of the given radius."""
    size = 2 * radius + 1
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    mask = (x * x + y * y) <= radius * radius
    return mask.astype(np.uint8)


def rolling_ball_background_subtraction_2d(
    plane: np.ndarray,
    radius: int = 50,
    use_gpu: bool = False,
) -> np.ndarray:
    """Apply rolling ball background subtraction to a single 2D plane.

    Uses grey erosion with a disk structuring element, then subtracts
    the erosion (background estimate) from the original.
    """
    footprint = _make_disk_footprint(radius)

    if use_gpu:
        gpu = _get_cupy_ndimage()
        if gpu is not None:
            cupy, cupy_ndi = gpu
            try:
                plane_gpu = cupy.asarray(plane)
                fp_gpu = cupy.asarray(footprint)
                bg = cupy_ndi.grey_erosion(plane_gpu, footprint=fp_gpu)
                bg = cupy_ndi.grey_dilation(bg, footprint=fp_gpu)
                result = cupy.clip(plane_gpu.astype(cupy.float32) - bg.astype(cupy.float32), 0, None)
                return result.astype(plane.dtype).get()
            except Exception:
                logger.debug("GPU rolling ball failed, falling back to CPU")

    bg = scipy.ndimage.grey_erosion(plane, footprint=footprint)
    bg = scipy.ndimage.grey_dilation(bg, footprint=footprint)
    result = np.clip(plane.astype(np.float32) - bg.astype(np.float32), 0, None)
    return result.astype(plane.dtype)


def rolling_ball_background_subtraction_3d(
    data: np.ndarray,
    radius: int = 50,
    use_gpu: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Apply rolling ball background subtraction to a 3D volume, per-slice."""
    result = np.empty_like(data)
    nz = data.shape[0]
    for z in range(nz):
        if cancel_check and cancel_check():
            return result
        result[z] = rolling_ball_background_subtraction_2d(data[z], radius, use_gpu)
        if progress_callback:
            progress_callback(z + 1, nz)
    return result


def gaussian_blur_2d(
    plane: np.ndarray,
    sigma: float = 1.0,
    use_gpu: bool = False,
) -> np.ndarray:
    """Apply Gaussian blur to a single 2D plane."""
    if use_gpu:
        gpu = _get_cupy_ndimage()
        if gpu is not None:
            cupy, cupy_ndi = gpu
            try:
                plane_gpu = cupy.asarray(plane.astype(np.float32))
                blurred = cupy_ndi.gaussian_filter(plane_gpu, sigma=sigma)
                return blurred.astype(plane.dtype).get()
            except Exception:
                logger.debug("GPU gaussian failed, falling back to CPU")

    blurred = scipy.ndimage.gaussian_filter(plane.astype(np.float32), sigma=sigma)
    return blurred.astype(plane.dtype)


def gaussian_blur_3d(
    data: np.ndarray,
    sigma: float = 1.0,
    use_gpu: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Apply Gaussian blur to a 3D volume, per-slice."""
    result = np.empty_like(data)
    nz = data.shape[0]
    for z in range(nz):
        if cancel_check and cancel_check():
            return result
        result[z] = gaussian_blur_2d(data[z], sigma, use_gpu)
        if progress_callback:
            progress_callback(z + 1, nz)
    return result


def unsharp_mask_2d(
    plane: np.ndarray,
    sigma: float = 1.0,
    mask_weight: float = 0.6,
    use_gpu: bool = False,
) -> np.ndarray:
    """Apply unsharp mask to a single 2D plane.

    Formula: result = (plane - weight * blurred) / (1 - weight)
    Matches ImageJ's unsharp mask formula.
    """
    if use_gpu:
        gpu = _get_cupy_ndimage()
        if gpu is not None:
            cupy, cupy_ndi = gpu
            try:
                plane_gpu = cupy.asarray(plane.astype(cupy.float32))
                blurred = cupy_ndi.gaussian_filter(plane_gpu, sigma=sigma)
                result = (plane_gpu - mask_weight * blurred) / (1.0 - mask_weight)
                if plane.dtype == np.uint16:
                    result = cupy.clip(result, 0, 65535)
                elif plane.dtype == np.uint8:
                    result = cupy.clip(result, 0, 255)
                else:
                    result = cupy.clip(result, 0, None)
                return result.astype(plane.dtype).get()
            except Exception:
                logger.debug("GPU unsharp mask failed, falling back to CPU")

    plane_f = plane.astype(np.float32)
    blurred = scipy.ndimage.gaussian_filter(plane_f, sigma=sigma)
    result = (plane_f - mask_weight * blurred) / (1.0 - mask_weight)
    if plane.dtype == np.uint16:
        result = np.clip(result, 0, 65535)
    elif plane.dtype == np.uint8:
        result = np.clip(result, 0, 255)
    else:
        result = np.clip(result, 0, None)
    return result.astype(plane.dtype)


def unsharp_mask_3d(
    data: np.ndarray,
    sigma: float = 1.0,
    mask_weight: float = 0.6,
    use_gpu: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Apply unsharp mask to a 3D volume, per-slice."""
    result = np.empty_like(data)
    nz = data.shape[0]
    for z in range(nz):
        if cancel_check and cancel_check():
            return result
        result[z] = unsharp_mask_2d(data[z], sigma, mask_weight, use_gpu)
        if progress_callback:
            progress_callback(z + 1, nz)
    return result


def apply_pipeline(
    data: np.ndarray,
    steps: list[dict],
    use_gpu: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Apply a sequence of processing steps to a 3D volume.

    Parameters
    ----------
    data : np.ndarray
        Input volume (Z, Y, X).
    steps : list[dict]
        Each step is a dict with keys:
        - "type": "rolling_ball" | "gaussian" | "unsharp_mask"
        - "enabled": bool
        - "params": dict of parameters for the filter
    use_gpu : bool
        Whether to attempt GPU acceleration.
    progress_callback : callable, optional
        Called with (steps_done, total_steps) after each step completes.
    cancel_check : callable, optional
        Return True to abort.

    Returns
    -------
    np.ndarray
        Processed volume, same shape and dtype as input.
    """
    enabled_steps = [s for s in steps if s.get("enabled", True)]
    total = len(enabled_steps)
    result = data.copy()

    for i, step in enumerate(enabled_steps):
        if cancel_check and cancel_check():
            return result

        step_type = step["type"]
        params = step.get("params", {})

        if step_type == "rolling_ball":
            result = rolling_ball_background_subtraction_3d(
                result,
                radius=params.get("radius", 50),
                use_gpu=use_gpu,
                cancel_check=cancel_check,
            )
        elif step_type == "gaussian":
            result = gaussian_blur_3d(
                result,
                sigma=params.get("sigma", 1.0),
                use_gpu=use_gpu,
                cancel_check=cancel_check,
            )
        elif step_type == "unsharp_mask":
            result = unsharp_mask_3d(
                result,
                sigma=params.get("sigma", 1.0),
                mask_weight=params.get("mask_weight", 0.6),
                use_gpu=use_gpu,
                cancel_check=cancel_check,
            )
        else:
            logger.warning("Unknown processing step type: %s", step_type)

        if progress_callback:
            progress_callback(i + 1, total)

    return result
