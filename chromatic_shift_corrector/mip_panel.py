"""Max intensity projection (MIP) panel with L-shaped orthogonal views.

Computes MIPs along Z, Y, and X axes and assembles them into an
ImageJ-style ortho layout.  Each channel is returned as a separate
grayscale panel so that napari can composite them with additive
blending, allowing individual channel toggling and contrast adjustment.

    +--------+------+
    |   XY   |  YZ  |
    | (maxZ) |(maxX)|
    +--------+------+
    |   XZ   |      |
    | (maxY) |      |
    +--------+------+
"""

from __future__ import annotations

import numpy as np


def compute_mips(
    volume: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute max intensity projections along each axis.

    Parameters
    ----------
    volume : np.ndarray
        3D array of shape (Z, Y, X).

    Returns
    -------
    mip_xy : np.ndarray
        Max along Z, shape (Y, X).
    mip_xz : np.ndarray
        Max along Y, shape (Z, X).
    mip_yz : np.ndarray
        Max along X then transposed, shape (Y, Z) so that the Y axis
        aligns vertically with the XY view.
    """
    mip_xy = volume.max(axis=0)      # (Y, X)
    mip_xz = volume.max(axis=1)      # (Z, X)
    mip_yz = volume.max(axis=2).T    # (Z, Y) -> (Y, Z)
    return mip_xy, mip_xz, mip_yz


def assemble_channel_panel(
    mip_xy: np.ndarray,
    mip_xz: np.ndarray,
    mip_yz: np.ndarray,
    gap: int = 2,
) -> np.ndarray:
    """Arrange single-channel MIPs into an L-shaped panel.

    All inputs must be 2D arrays of the same dtype.  The gap region is
    zero-filled.

    Parameters
    ----------
    mip_xy : (ny, nx)
    mip_xz : (nz, nx)
    mip_yz : (ny, nz)
    gap : int
        Dark-pixel gap between sub-panels.

    Returns
    -------
    np.ndarray
        2D array of shape (ny + gap + nz, nx + gap + nz).
    """
    ny, nx = mip_xy.shape
    nz = mip_xz.shape[0]
    nz_yz = mip_yz.shape[1]

    h = ny + gap + nz
    w = nx + gap + nz_yz

    panel = np.zeros((h, w), dtype=mip_xy.dtype)
    panel[:ny, :nx] = mip_xy
    panel[:ny, nx + gap : nx + gap + nz_yz] = mip_yz
    panel[ny + gap : ny + gap + nz, :nx] = mip_xz
    return panel


def build_crosshair_overlay(
    ny: int,
    nx: int,
    nz: int,
    center_y: int,
    center_x: int,
    center_z: int,
    gap: int = 2,
    line_value: float = 1.0,
) -> np.ndarray:
    """Build a grayscale crosshair overlay for the L-shaped panel.

    Returns a float32 2D array with bright crosshair lines on a black
    background, suitable for display as an additive-blended layer on top
    of the per-channel MIP panels.

    Parameters
    ----------
    ny, nx, nz : int
        Sub-volume dimensions (must match the channel panels).
    center_y, center_x, center_z : int
        Crosshair positions in sub-volume coordinates.
    gap : int
        Gap between sub-panels (must match the channel panels).
    line_value : float
        Intensity of the crosshair lines.
    """
    h = ny + gap + nz
    w = nx + gap + nz

    overlay = np.zeros((h, w), dtype=np.float32)

    # XY panel (top-left)
    if 0 <= center_y < ny:
        overlay[center_y, :nx] = line_value
    if 0 <= center_x < nx:
        overlay[:ny, center_x] = line_value

    # XZ panel (bottom-left, offset by ny + gap)
    yz_off = ny + gap
    if 0 <= center_z < nz:
        overlay[yz_off + center_z, :nx] = line_value
    if 0 <= center_x < nx:
        overlay[yz_off : yz_off + nz, center_x] = line_value

    # YZ panel (top-right, offset by nx + gap)
    xz_off = nx + gap
    if 0 <= center_y < ny:
        overlay[center_y, xz_off : xz_off + nz] = line_value
    if 0 <= center_z < nz:
        overlay[:ny, xz_off + center_z] = line_value

    return overlay
