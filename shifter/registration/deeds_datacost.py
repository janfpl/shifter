"""Discrete per-control-point data cost for deformable registration.

Port of ``dataCostCL`` (``dataCostD.h``). For every control point and every
candidate displacement label in the ``(2·hw+1)^3`` cube, accumulate the
descriptor dissimilarity over that control point's ``step^3`` block. As
elsewhere in this port (see ``deeds.py``), the MIND-SSC descriptor is the
float32 ``exp(-x)`` form and dissimilarity is **SSD** rather than the reference's
64-bit quantised popcount Hamming — the one deliberate faithfulness compromise.

The cost cube for a control point is ``(L, L, L)`` with axes ``(dz, dy, dx)`` and
label index ``a`` meaning displacement ``(a - hw)·quant`` — matching the label
convention used by :mod:`shifter.registration.deeds_mst`.
"""

from __future__ import annotations

import numpy as np


def _skip_for(step: int, randnum: int) -> int:
    """Within-block subsampling stride that also divides *step*.

    Keeping ``step % skip == 0`` means the within-block stride-*skip* sampling is
    equivalent to a global stride-*skip* subsample (block borders fall on
    multiples of *step*), which lets the SSD be computed only at the sampled
    voxels. Follows the spirit of the reference's speed schedule.
    """
    if randnum > 0 and step % 2 == 0:
        return 2
    return 1


def data_cost(
    desc_fixed,
    desc_moving,
    step: int,
    hw: int,
    quant: int,
    alpha: float,
    grid_shape,
    xp=np,
    randnum: int = 1,
    data_scale: float = 1.0,
):
    """Build the discrete data-cost tensor ``costall[num_cp, L, L, L]``.

    Parameters
    ----------
    desc_fixed, desc_moving : (12, Z, Y, X) float32
        MIND-SSC descriptors of the fixed and (already field-warped) moving
        volumes.
    step, hw, quant, alpha : level parameters.
    grid_shape : (gz, gy, gx) control-grid size (``Z//step`` etc.).
    randnum : selects the within-block subsampling stride (speed vs. accuracy).
    data_scale : multiplies the cost, to rebalance SSD magnitude against the
        regularizer relative to the reference's popcount scale.

    Returns a ``(num_cp, L, L, L)`` float32 array (num_cp = prod(grid_shape)),
    cost cube axes ``(dz, dy, dx)``.
    """
    L = 2 * hw + 1
    gz, gy, gx = grid_shape
    num_cp = gz * gy * gx
    quant = int(quant)
    pad = hw * quant
    skip = _skip_for(step, randnum)

    desc_fixed = xp.asarray(desc_fixed, dtype=xp.float32)
    desc_moving = xp.asarray(desc_moving, dtype=xp.float32)
    cz, cy, cx = gz * step, gy * step, gx * step
    sb = step // skip  # samples per block per axis

    # Edge-pad the moving descriptor so any centred label displacement is a plain
    # slice: moving(p + (a-hw)·quant) == descm_pad[:, a·quant : a·quant + size].
    descm_pad = xp.pad(
        desc_moving, ((0, 0), (pad, pad), (pad, pad), (pad, pad)), mode="edge"
    )
    # Fixed descriptor sampled at the block-sample voxels (global stride `skip`,
    # valid because step % skip == 0).
    fixed_s = desc_fixed[:, :cz:skip, :cy:skip, :cx:skip]

    # Data-vs-regularizer weight. The reference sums the popcount over the
    # sampled block voxels and divides once by maxsamp (== the sample count) via
    # alpha1, i.e. a mean over samples. We take that mean directly with
    # ``.mean(...)`` below, so alpha1 must NOT divide by maxsamp again — doing so
    # would underweight the data term by (step/skip)^3, and by a *different*
    # factor at every level (since skip varies), breaking the reference's uniform
    # balance. ``data_scale`` then rescales the SSD magnitude to the popcount
    # range ALPHA was tuned for.
    alpha1 = 0.5 * (step / (alpha * quant)) * data_scale

    costall = xp.empty((num_cp, L, L, L), dtype=xp.float32)
    for a0 in range(L):
        oz = a0 * quant
        for a1 in range(L):
            oy = a1 * quant
            for a2 in range(L):
                ox = a2 * quant
                # Moving sampled at (sample_pos + label displacement).
                sub = descm_pad[:, oz:oz + cz:skip, oy:oy + cy:skip, ox:ox + cx:skip]
                diff = fixed_s - sub
                ssd = xp.einsum("czyx,czyx->zyx", diff, diff)
                grid = ssd.reshape(gz, sb, gy, sb, gx, sb).mean(axis=(1, 3, 5))
                costall[:, a0, a1, a2] = grid.reshape(-1) * alpha1

    return costall
