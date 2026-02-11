"""Orthogonal viewing utilities.

napari natively supports switching between XY, XZ, and YZ slice orientations
via its dims controls. This module provides helper functions for programmatic
access to those controls if needed by the widget or scripts.
"""

from __future__ import annotations

from typing import Any


def set_view_xy(viewer: Any) -> None:
    """Set the napari viewer to the default XY orientation.

    This is the standard view where:
      - displayed axes are Y (vertical) and X (horizontal)
      - the slider controls Z
    """
    ndim = viewer.dims.ndim
    if ndim >= 3:
        viewer.dims.order = tuple(range(ndim))  # (0, 1, 2, ...) → Z is dim 0


def set_view_xz(viewer: Any) -> None:
    """Set the napari viewer to XZ orientation.

    Displayed axes become Z (vertical) and X (horizontal); the slider
    controls Y.
    """
    ndim = viewer.dims.ndim
    if ndim >= 3:
        # Put Y (dim 1) first so it becomes the slider axis.
        order = list(range(ndim))
        order[0], order[1] = 1, 0  # swap Z and Y
        viewer.dims.order = tuple(order)


def set_view_yz(viewer: Any) -> None:
    """Set the napari viewer to YZ orientation.

    Displayed axes become Z (vertical) and Y (horizontal); the slider
    controls X.
    """
    ndim = viewer.dims.ndim
    if ndim >= 3:
        # Put X (dim 2) first so it becomes the slider axis.
        order = list(range(ndim))
        order[0], order[2] = 2, 0  # swap Z and X
        viewer.dims.order = tuple(order)
