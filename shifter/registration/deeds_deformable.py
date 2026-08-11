"""Full deformable deedsBCV registration (numpy, optional cupy).

Solves a dense sub-voxel displacement field with the faithful reference
algorithm: a multi-resolution control-point grid, MIND-SSC descriptors, a
discrete per-control-point data cost, minimum-spanning-tree belief-propagation
regularization, and forward+backward symmetric (inverse-consistent) composition.
Ports ``deedsBCV0.cpp``'s per-level driver (levels 222-330).

Scope (v1): produces the field and the **warped corrected volume** for an
in-RAM ROI / downsampled volume. This is deedsBCV's deformable core in a
translation-and-deformation role but without the affine pre-stage (``linearBCV``)
— it assumes roughly pre-aligned inputs. The descriptor/data-cost deviation
(float32 ``exp(-x)`` MIND-SSC + SSD instead of quantised popcount Hamming) is
inherited from :mod:`shifter.registration.deeds`.

Field convention: ``corrected(p) = moving(p + field(p))`` — the field, applied
by :func:`shifter.registration.deeds_field.warp_volume`, warps moving onto
fixed. (Note this is the negation of the *reported* integer shift the
translation methods return, but it is the field the warp consumes directly.)

The regularizer and MST run on the host (numpy); descriptors, data cost and
warping use ``xp`` (cupy when ``use_gpu`` and available, else numpy).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from shifter.registration.base import ProgressCallback
from shifter.registration.deeds import mind_ssc
from shifter.registration.deeds_datacost import data_cost
from shifter.registration.deeds_field import (
    upsample_field,
    warp_volume,
    compose_consistent,
)
from shifter.registration.deeds_mst import prims_graph, regularise

logger = logging.getLogger(__name__)

ALGORITHM_NAME = "deedsBCV (deformable)"

# Reference parameter schedule (deedsBCV0.cpp:76-79). Coarsest level first.
GRID_SPACING = (8, 7, 6, 5, 4)
SEARCH_RADIUS = (8, 7, 6, 5, 4)
QUANTISATION = (5, 4, 3, 2, 1)
MIND_STEP = (3, 3, 2, 2, 1)
ALPHA = 1.6

# Rebalances the float32-SSD data cost against the squared-label regularizer,
# since our descriptor SSD is a different magnitude than the reference popcount
# Hamming that ALPHA was tuned for. Chosen to minimise endpoint error against a
# known field on *noisy* synthetic data (the min of the data-vs-smoothness
# trade-off — larger values start fitting noise), which coincides with a strong
# SSD reduction and a fold-free field.
DATA_SCALE = 8.0


@dataclass
class DeformableResult:
    """Result of a deformable registration.

    Attributes
    ----------
    field : (3, gz, gy, gx) float32
        Displacement field at the finest control-grid resolution, ``(dz, dy, dx)``
        in absolute (full-resolution) voxels. Apply with ``warp_volume`` after
        ``upsample_field`` to the volume resolution.
    volume_shape : (Z, Y, X)
        Shape the field warps.
    algorithm_name : str
    """

    field: np.ndarray
    volume_shape: tuple[int, int, int]
    algorithm_name: str = ALGORITHM_NAME


def _levels(levels_params):
    if levels_params is not None:
        return levels_params
    return list(zip(GRID_SPACING, SEARCH_RADIUS, QUANTISATION, MIND_STEP))


def register_deformable(
    reference_volume: np.ndarray,
    moving_volume: np.ndarray,
    *,
    use_gpu: bool = False,
    levels_params=None,
    data_scale: float = DATA_SCALE,
    alpha: float = ALPHA,
    progress_callback: ProgressCallback | None = None,
) -> DeformableResult:
    """Solve the deformable field aligning *moving* to *reference*.

    Returns a :class:`DeformableResult`; use :func:`warp_corrected` to obtain the
    corrected volume.
    """
    xp = np
    if use_gpu:
        try:
            import cupy as cp

            cp.zeros(1)  # probe
            xp = cp
        except Exception as exc:  # pragma: no cover - depends on hardware
            logger.warning("GPU deformable unavailable (%s); using CPU", exc)
            xp = np

    fixed = xp.asarray(reference_volume, dtype=xp.float32)
    moving = xp.asarray(moving_volume, dtype=xp.float32)
    shape = tuple(int(s) for s in fixed.shape)
    if fixed.shape != moving.shape:
        raise ValueError(
            f"reference and moving volumes must match: {shape} vs "
            f"{tuple(int(s) for s in moving.shape)}"
        )
    Z, Y, X = shape

    levels = _levels(levels_params)
    n_levels = len(levels)

    # The coarsest control grid needs at least one cell per axis, plus a margin
    # so the descriptor and block reduction have data to work with. Below this a
    # zero-size grid would fail obscurely inside prims_graph/regularise.
    coarsest_step = max(step for step, *_ in levels)
    min_size = 2 * coarsest_step
    if min(shape) < min_size:
        raise ValueError(
            f"deformable registration needs a volume of at least {min_size} "
            f"voxels per axis (coarsest grid spacing {coarsest_step}); got {shape}. "
            "Enlarge the ROI or its Z range."
        )

    def grid_of(step):
        return (Z // step, Y // step, X // step)

    # Forward (uf) and backward (ub) control-grid fields, initialised at the
    # coarsest grid as zeros.
    g0 = grid_of(levels[0][0])
    uf = xp.zeros((3,) + g0, dtype=xp.float32)
    ub = xp.zeros((3,) + g0, dtype=xp.float32)

    desc_cache: dict[int, tuple] = {}

    def descriptors(mstep):
        if mstep not in desc_cache:
            desc_cache[mstep] = (mind_ssc(fixed, mstep, xp), mind_ssc(moving, mstep, xp))
        return desc_cache[mstep]

    def to_host(a):
        return a.get() if hasattr(a, "get") else np.asarray(a)

    for li, (step, hw, quant, mstep) in enumerate(levels):
        g = grid_of(step)
        desc_fix, desc_mov = descriptors(mstep)

        # ---- forward: warp moving by current field, cost vs fixed ----------
        u0 = upsample_field(uf, g, xp)
        warped_mov = warp_volume(moving, upsample_field(u0, shape, xp), xp)
        cost = data_cost(
            desc_fix, mind_ssc(warped_mov, mstep, xp),
            step, hw, quant, alpha, g, xp, data_scale=data_scale,
        )
        order, parents, edgemst = prims_graph(to_host(fixed), step, g)
        uf = xp.asarray(
            regularise(to_host(cost), to_host(u0), order, parents, edgemst, hw, quant, g),
            dtype=xp.float32,
        )

        # ---- backward: warp fixed by current backward field, cost vs moving -
        u0b = upsample_field(ub, g, xp)
        warped_fix = warp_volume(fixed, upsample_field(u0b, shape, xp), xp)
        cost_b = data_cost(
            desc_mov, mind_ssc(warped_fix, mstep, xp),
            step, hw, quant, alpha, g, xp, data_scale=data_scale,
        )
        order_b, parents_b, edgemst_b = prims_graph(to_host(moving), step, g)
        ub = xp.asarray(
            regularise(to_host(cost_b), to_host(u0b), order_b, parents_b, edgemst_b, hw, quant, g),
            dtype=xp.float32,
        )

        # ---- inverse-consistent symmetric composition ---------------------
        uf, ub = compose_consistent(uf, ub, step, xp)

        if progress_callback is not None:
            progress_callback((li + 1) / n_levels)

    return DeformableResult(field=to_host(uf), volume_shape=shape)


def warp_corrected(moving_volume, result: DeformableResult, use_gpu: bool = False):
    """Warp *moving_volume* by the solved field, returning the corrected volume."""
    xp = np
    if use_gpu:
        try:
            import cupy as cp

            cp.zeros(1)
            xp = cp
        except Exception:  # pragma: no cover
            xp = np
    field_full = upsample_field(xp.asarray(result.field), result.volume_shape, xp)
    warped = warp_volume(xp.asarray(moving_volume), field_full, xp, order=1)
    out = warped.get() if hasattr(warped, "get") else np.asarray(warped)
    # Round (not truncate) when writing back to an integer dtype, so trilinear
    # sampling does not introduce a systematic ~0.5-intensity downward bias.
    dtype = np.dtype(moving_volume.dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        out = np.clip(np.rint(out), info.min, info.max)
    return out.astype(dtype)
