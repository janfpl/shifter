"""Max intensity projection (MIP) panel with L-shaped orthogonal views.

Computes MIPs along Z, Y, and X axes, applies per-channel colormaps,
composites them additively into a single RGB image, and assembles
the three views into an ImageJ-style ortho layout:

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

# Simple single-color LUTs for common napari colormaps.
_COLORMAP_RGB: dict[str, np.ndarray] = {
    "green": np.array([0.0, 1.0, 0.0]),
    "magenta": np.array([1.0, 0.0, 1.0]),
    "cyan": np.array([0.0, 1.0, 1.0]),
    "yellow": np.array([1.0, 1.0, 0.0]),
    "red": np.array([1.0, 0.0, 0.0]),
    "blue": np.array([0.0, 0.0, 1.0]),
    "gray": np.array([1.0, 1.0, 1.0]),
}


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


def normalize_intensity(
    arr: np.ndarray,
    plow: float = 0.1,
    phigh: float = 99.9,
) -> np.ndarray:
    """Normalize to [0, 1] using percentile-based contrast limits."""
    low = np.percentile(arr, plow)
    high = np.percentile(arr, phigh)
    if high <= low:
        return np.zeros_like(arr, dtype=np.float32)
    result = (arr.astype(np.float32) - low) / (high - low)
    return np.clip(result, 0.0, 1.0)


def apply_colormap(mip: np.ndarray, colormap: str) -> np.ndarray:
    """Apply a named colormap to a normalized 2D array, producing RGB.

    Parameters
    ----------
    mip : np.ndarray
        Normalized 2D float32 array in [0, 1], shape (H, W).
    colormap : str
        Colormap name (e.g. ``"green"``, ``"magenta"``, ``"viridis"``).

    Returns
    -------
    np.ndarray
        RGB image, shape (H, W, 3), float32 in [0, 1].
    """
    if colormap in _COLORMAP_RGB:
        return mip[..., np.newaxis] * _COLORMAP_RGB[colormap]

    # Fall back to matplotlib for complex colormaps (viridis, inferno, …).
    try:
        from matplotlib import colormaps

        cmap = colormaps[colormap]
        rgba = cmap(mip)  # (H, W, 4)
        return rgba[..., :3].astype(np.float32)
    except (ImportError, KeyError):
        return mip[..., np.newaxis] * np.array([1.0, 1.0, 1.0])


def composite_rgb(rgb_layers: list[np.ndarray]) -> np.ndarray:
    """Additively composite multiple RGB images, clamped to [0, 1]."""
    result = np.zeros_like(rgb_layers[0])
    for layer in rgb_layers:
        result += layer
    return np.clip(result, 0.0, 1.0)


def assemble_panel(
    mip_xy: np.ndarray,
    mip_xz: np.ndarray,
    mip_yz: np.ndarray,
    gap: int = 2,
) -> np.ndarray:
    """Arrange three composited MIP views into an L-shaped panel.

    Layout (see module docstring for diagram).

    Parameters
    ----------
    mip_xy : (ny, nx, 3)
    mip_xz : (nz, nx, 3)
    mip_yz : (ny, nz, 3)
    gap : int
        Dark-pixel gap between panels.
    """
    ny, nx = mip_xy.shape[:2]
    nz = mip_xz.shape[0]
    nz_yz = mip_yz.shape[1]

    h = ny + gap + nz
    w = nx + gap + nz_yz

    panel = np.zeros((h, w, 3), dtype=np.float32)
    panel[:ny, :nx] = mip_xy
    panel[:ny, nx + gap : nx + gap + nz_yz] = mip_yz
    panel[ny + gap : ny + gap + nz, :nx] = mip_xz
    return panel


def draw_crosshairs(
    panel: np.ndarray,
    ny: int,
    nx: int,
    nz: int,
    center_y: int,
    center_x: int,
    center_z: int,
    gap: int = 2,
    color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    alpha: float = 0.5,
) -> np.ndarray:
    """Draw crosshair lines on the L-shaped panel.

    Crosshairs indicate the orthogonal slice position on each view.
    """
    out = panel.copy()
    cc = np.array(color, dtype=np.float32)

    def _blend_row(row: int, col_start: int, col_end: int) -> None:
        out[row, col_start:col_end] = (
            alpha * cc + (1 - alpha) * out[row, col_start:col_end]
        )

    def _blend_col(col: int, row_start: int, row_end: int) -> None:
        out[row_start:row_end, col] = (
            alpha * cc + (1 - alpha) * out[row_start:row_end, col]
        )

    # XY panel (top-left): crosshairs at (center_y, center_x)
    if 0 <= center_y < ny:
        _blend_row(center_y, 0, nx)
    if 0 <= center_x < nx:
        _blend_col(center_x, 0, ny)

    # XZ panel (bottom-left, offset by ny + gap): crosshairs at (center_z, center_x)
    yz_off = ny + gap
    if 0 <= center_z < nz:
        _blend_row(yz_off + center_z, 0, nx)
    if 0 <= center_x < nx:
        _blend_col(center_x, yz_off, yz_off + nz)

    # YZ panel (top-right, offset by nx + gap): crosshairs at (center_y, center_z)
    xz_off = nx + gap
    if 0 <= center_y < ny:
        _blend_row(center_y, xz_off, xz_off + nz)
    if 0 <= center_z < nz:
        _blend_col(xz_off + center_z, 0, ny)

    return np.clip(out, 0.0, 1.0)


def build_mip_panel(
    volumes: list[np.ndarray],
    colormaps: list[str],
    crosshair_zyx: tuple[int, int, int] | None = None,
    gap: int = 2,
) -> np.ndarray:
    """Build a complete MIP panel from multiple channel volumes.

    Parameters
    ----------
    volumes : list of np.ndarray
        Shifted 3D sub-volumes for each channel, each (Z, Y, X).
    colormaps : list of str
        Colormap name per channel.
    crosshair_zyx : (z, y, x) or None
        Position for crosshair lines in sub-volume coordinates.
        Defaults to the centre of the volume.
    gap : int
        Dark-pixel gap between views.

    Returns
    -------
    np.ndarray
        L-shaped RGB panel, shape (H, W, 3), float32 in [0, 1].
    """
    nz, ny, nx = volumes[0].shape

    xy_layers: list[np.ndarray] = []
    xz_layers: list[np.ndarray] = []
    yz_layers: list[np.ndarray] = []

    for vol, cmap in zip(volumes, colormaps):
        mip_xy, mip_xz, mip_yz = compute_mips(vol)
        xy_layers.append(apply_colormap(normalize_intensity(mip_xy), cmap))
        xz_layers.append(apply_colormap(normalize_intensity(mip_xz), cmap))
        yz_layers.append(apply_colormap(normalize_intensity(mip_yz), cmap))

    panel = assemble_panel(
        composite_rgb(xy_layers),
        composite_rgb(xz_layers),
        composite_rgb(yz_layers),
        gap=gap,
    )

    if crosshair_zyx is None:
        crosshair_zyx = (nz // 2, ny // 2, nx // 2)
    cz, cy, cx = crosshair_zyx
    return draw_crosshairs(panel, ny, nx, nz, cy, cx, cz, gap=gap)


def build_mip_panel_split(
    volumes: list[np.ndarray],
    colormaps: list[str],
    gap: int = 2,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Build MIP panel *without* crosshairs and return sub-volume dims.

    Returns ``(panel, (nz, ny, nx))`` so that crosshairs can be drawn
    separately (e.g. in response to slider changes) without recomputing
    the projections.
    """
    nz, ny, nx = volumes[0].shape

    xy_layers: list[np.ndarray] = []
    xz_layers: list[np.ndarray] = []
    yz_layers: list[np.ndarray] = []

    for vol, cmap in zip(volumes, colormaps):
        mip_xy, mip_xz, mip_yz = compute_mips(vol)
        xy_layers.append(apply_colormap(normalize_intensity(mip_xy), cmap))
        xz_layers.append(apply_colormap(normalize_intensity(mip_xz), cmap))
        yz_layers.append(apply_colormap(normalize_intensity(mip_yz), cmap))

    panel = assemble_panel(
        composite_rgb(xy_layers),
        composite_rgb(xz_layers),
        composite_rgb(yz_layers),
        gap=gap,
    )
    return panel, (nz, ny, nx)
