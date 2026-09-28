"""Dense displacement-field primitives for deformable registration.

A displacement field is a single ``(3, gz, gy, gx)`` float32 array whose leading
axis is ``(dz, dy, dx)`` — the same ``(Z, Y, X)`` ordering the rest of the repo
uses for volumes. Displacements are **absolute voxels** (of the full-resolution
volume), and the warping convention matches the translation methods:

    corrected(p) = moving(p + field(p))

i.e. the field maps an output/reference coordinate to the moving coordinate it
should sample. This mirrors deedsBCV's ``interp3(..., flag=true)`` (add grid
coordinate) and ``warpImageCL``.

These are ports of the geometry helpers in the reference ``transformations.h``:
``interp3`` (trilinear), ``upsampleDeformationsCL`` (resample the control-grid
field, no magnitude rescale), and ``consistentMappingCL`` (inverse-consistent
symmetric composition). All work on numpy or cupy via the ``xp`` argument.
"""

from __future__ import annotations

import numpy as np


def _ndimage_module(xp):
    """Return the ndimage module matching array module *xp* (numpy or cupy)."""
    if xp is np:
        import scipy.ndimage as ndi

        return ndi
    import cupyx.scipy.ndimage as ndi

    return ndi


def _identity_grid(shape: tuple[int, int, int], xp):
    """Return a ``(3, Z, Y, X)`` array of the voxel's own (z, y, x) index."""
    zz, yy, xx = xp.meshgrid(
        xp.arange(shape[0], dtype=xp.float32),
        xp.arange(shape[1], dtype=xp.float32),
        xp.arange(shape[2], dtype=xp.float32),
        indexing="ij",
    )
    return xp.stack([zz, yy, xx])


def warp_volume(volume, field, xp=np, order: int = 1):
    """Warp *volume* by *field*: ``out(p) = volume(p + field(p))``.

    Parameters
    ----------
    volume : (Z, Y, X) ndarray
    field : (3, Z, Y, X) ndarray
        Displacement field at the volume's resolution, ``(dz, dy, dx)``.
    order : int
        Interpolation order — 1 (trilinear) for images, 0 for label maps.

    Out-of-bounds samples replicate the border (``mode="nearest"``), matching
    the clamped indexing in the reference ``interp3``.
    """
    ndi = _ndimage_module(xp)
    vol = xp.asarray(volume, dtype=xp.float32)
    coords = _identity_grid(vol.shape, xp) + xp.asarray(field, dtype=xp.float32)
    return ndi.map_coordinates(vol, coords, order=order, mode="nearest")


def upsample_field(field, out_shape: tuple[int, int, int], xp=np):
    """Trilinearly resample a displacement *field* to *out_shape*.

    Displacement magnitudes are **not** rescaled when the grid resolution
    changes — they stay in absolute (full-resolution) voxel units, matching
    ``upsampleDeformationsCL`` (whose rescale block is commented out).
    """
    ndi = _ndimage_module(xp)
    field = xp.asarray(field, dtype=xp.float32)
    in_shape = field.shape[1:]

    # Input coordinate for each output index: out_index * (in_size / out_size),
    # exactly as the reference builds x1 = j / (n / n2).
    axes = [
        xp.arange(out_shape[a], dtype=xp.float32) * (in_shape[a] / out_shape[a])
        for a in range(3)
    ]
    zz, yy, xx = xp.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    coords = xp.stack([zz, yy, xx])

    out = xp.empty((3,) + tuple(out_shape), dtype=xp.float32)
    for c in range(3):
        out[c] = ndi.map_coordinates(field[c], coords, order=1, mode="nearest")
    return out


def _compose(field_a, disp_b, xp):
    """Return ``field_a`` sampled at ``(identity + disp_b)`` — i.e. A∘(id+B)."""
    ndi = _ndimage_module(xp)
    coords = _identity_grid(field_a.shape[1:], xp) + disp_b
    out = xp.empty_like(field_a)
    for c in range(3):
        out[c] = ndi.map_coordinates(field_a[c], coords, order=1, mode="nearest")
    return out


def compose_consistent(forward, backward, factor: int, xp=np, iters: int = 10):
    """Make *forward* and *backward* fields inverse-consistent (symmetric).

    Port of ``consistentMappingCL``. Both fields are scaled into grid units
    (``1/factor``), refined for *iters* iterations of

        f ← 0.5·f − 0.5·(b ∘ (id + f))
        b ← 0.5·b − 0.5·(f ∘ (id + b))

    (both halves using the pre-update fields), then rescaled by *factor*.
    *factor* is the control-grid spacing that converts absolute-voxel
    displacements to grid-index units.

    Returns the updated ``(forward, backward)``.
    """
    inv = 1.0 / float(factor)
    f = xp.asarray(forward, dtype=xp.float32) * inv
    b = xp.asarray(backward, dtype=xp.float32) * inv

    for _ in range(iters):
        comp_b = _compose(b, f, xp)  # b ∘ (id + f)
        comp_f = _compose(f, b, xp)  # f ∘ (id + b)  (uses pre-update f)
        f = 0.5 * f - 0.5 * comp_b
        b = 0.5 * b - 0.5 * comp_f

    return f * float(factor), b * float(factor)
