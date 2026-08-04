"""deedsBCV-style registration using MIND-SSC self-similarity descriptors.

This is a numpy/cupy implementation of the image-similarity core of
**deedsBCV** (Mattias P. Heinrich, MIT-licensed,
https://github.com/mattiaspaul/deedsBCV): the MIND-SSC (modality-independent
neighbourhood descriptor with self-similarity context) descriptor from
``src/MINDSSCbox.h``, combined with a coarse-to-fine discrete displacement
search over a downsampling pyramid.

Scope
-----
deedsBCV proper solves a *deformable* registration: dense discrete
displacements on a control-point grid, regularised with a minimum-spanning-tree
of the grid. Shifter's registration interface produces a single global integer
translation per channel, so what is reproduced here is the descriptor and the
discrete data-cost search — effectively the translation-only stage of deeds
(the same role ``linearBCV`` plays before the deformable pass). The
regularisation and the deformable field are deliberately not implemented: there
is nowhere for a non-rigid field to go in this pipeline.

Why it is useful here
---------------------
MIND-SSC compares the *local self-similarity structure* of a patch rather than
its intensities, so it is insensitive to the intensity relationship between the
two volumes. That makes it a good fit for channels with very different
fluorophore responses — the same situation that motivates mutual information,
but evaluated with a fixed-cost descriptor comparison instead of an exhaustive
histogram search.

Differences from the reference C++
----------------------------------
* The reference quantises each of the 12 descriptor entries into a 64-bit
  integer and compares descriptors with a popcount Hamming distance. Here the
  descriptors are kept as ``float32`` (``exp(-x)``, the continuous form the
  reference leaves commented out in ``MINDSSCbox.h``) and compared with a sum
  of squared differences. Same ordering of candidates, no bit-packing.
* Box filtering uses ``scipy.ndimage.uniform_filter`` (a mean) where the
  reference uses a running-sum box filter; the constant factor cancels in the
  per-voxel noise normalisation that follows.
* The data cost is accumulated over a strided grid of sample points rather than
  every voxel, which is the same idea as the reference's control-point grid.

References
----------
Heinrich et al., "MRF-Based Deformable Registration and Ventilation Estimation
of Lung CT", IEEE TMI 32(7), 2013 (deeds).
Heinrich et al., "Towards Realtime Multimodal Fusion for Image-Guided
Interventions Using Self-Similarities", MICCAI 2013 (MIND-SSC).
"""

from __future__ import annotations

import logging
import math

import numpy as np

from shifter.registration.base import (
    ProgressCallback,
    RegistrationAlgorithm,
    RegistrationResult,
)

logger = logging.getLogger(__name__)

ALGORITHM_NAME = "deedsBCV (MIND-SSC)"

# The six self-similarity distance patches, as (dz, dy, dx) multiples of the
# quantisation step. Taken from the dx/dy/dz tables in deedsBCV's
# MINDSSCbox.h::descriptor (their i indexes Y, j indexes X, k indexes Z).
_SSC_DISTANCE_OFFSETS = (
    (0, 1, 1),
    (0, -1, 1),
    (1, 0, -1),
    (1, -1, 0),
    (1, 0, 1),
    (1, 1, 0),
)

# The twelve descriptor entries: each samples one of the six distance patches
# at a neighbouring voxel (the sx/sy/sz + index tables of the same function).
_SSC_PATCH_OFFSETS = (
    (0, 0, -1),
    (0, -1, 0),
    (0, 0, -1),
    (0, 1, 0),
    (-1, 0, 0),
    (0, 0, 1),
    (-1, 0, 0),
    (0, 1, 0),
    (-1, 0, 0),
    (0, 0, -1),
    (-1, 0, 0),
    (0, -1, 0),
)
_SSC_PATCH_SOURCE = (0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5)

_N_DESCRIPTORS = 12

# Downsampling factors of the coarse-to-fine pyramid, coarsest first. Levels
# whose downsampled volume would be smaller than _MIN_LEVEL_SIZE per axis are
# dropped (the full-resolution level is always kept).
_PYRAMID_FACTORS = (4, 2, 1)
_MIN_LEVEL_SIZE = 16

# Cap on the number of voxel positions used to accumulate the data cost. Higher
# means a more stable cost surface and a slower search.
_MAX_SAMPLES = 20000

# Element budget for one gathered candidate batch, which sets how many
# candidate shifts are evaluated per vectorised step (and therefore how often
# progress is reported).
_BATCH_ELEMENTS = 8_000_000


def _report(progress_callback: ProgressCallback | None, fraction: float) -> None:
    """Call *progress_callback* with a clamped fraction in [0, 1], if provided."""
    if progress_callback is not None:
        progress_callback(min(1.0, max(0.0, fraction)))


def _shifted(vol, dz: int, dy: int, dx: int):
    """``out[z, y, x] = vol[z + dz, y + dy, x + dx]``, clamped at the borders.

    Out-of-bounds positions keep the voxel's own value, matching ``imshift`` in
    the reference implementation.
    """
    if dz == 0 and dy == 0 and dx == 0:
        return vol
    out = vol.copy()
    src = []
    dst = []
    for d, n in zip((dz, dy, dx), vol.shape):
        src.append(slice(max(d, 0), n + min(d, 0)))
        dst.append(slice(max(-d, 0), n + min(-d, 0)))
    out[tuple(dst)] = vol[tuple(src)]
    return out


def _ndimage_module(xp):
    """Return the ndimage module matching array module *xp*."""
    if xp is np:
        import scipy.ndimage as ndi

        return ndi
    import cupyx.scipy.ndimage as ndi

    return ndi


def mind_ssc(volume, quantisation_step: int = 1, xp=np):
    """Compute the MIND-SSC descriptor of *volume*.

    Parameters
    ----------
    volume : ndarray
        3-D array (Z, Y, X), any numeric dtype.
    quantisation_step : int
        Spacing ``qs`` of the self-similarity neighbourhood, in voxels. Also
        the half-width of the box filter, as in the reference.
    xp : module
        ``numpy`` or ``cupy``.

    Returns
    -------
    ndarray
        ``(12, Z, Y, X)`` float32 descriptor. Each voxel's 12 values are
        non-negative and scaled by the local noise estimate.
    """
    ndi = _ndimage_module(xp)
    qs = max(1, int(quantisation_step))

    vol = xp.asarray(volume, dtype=xp.float32)
    std = float(vol.std())
    if std > 1e-12:
        vol = (vol - vol.mean()) / std

    # ---- six patch distances, box-filtered -----------------------------
    distances = xp.empty((len(_SSC_DISTANCE_OFFSETS),) + vol.shape, dtype=xp.float32)
    for i, (dz, dy, dx) in enumerate(_SSC_DISTANCE_OFFSETS):
        diff = _shifted(vol, dz * qs, dy * qs, dx * qs) - vol
        diff *= diff
        distances[i] = ndi.uniform_filter(diff, size=2 * qs + 1, mode="nearest")

    # ---- twelve descriptor entries: distances sampled at neighbours ----
    desc = xp.empty((_N_DESCRIPTORS,) + vol.shape, dtype=xp.float32)
    for i, (src, (dz, dy, dx)) in enumerate(zip(_SSC_PATCH_SOURCE, _SSC_PATCH_OFFSETS)):
        desc[i] = _shifted(distances[src], dz * qs, dy * qs, dx * qs)

    # ---- per-voxel normalisation by the local noise estimate -----------
    desc -= desc.min(axis=0, keepdims=True)
    noise = xp.maximum(desc.mean(axis=0, keepdims=True), 1e-6)
    desc /= noise
    return xp.exp(-desc)


def _downsample(volume, factor: int, xp=np):
    """Block-mean downsample by an integer *factor* (trailing voxels dropped)."""
    if factor <= 1:
        return volume
    nz, ny, nx = volume.shape
    cz, cy, cx = (nz // factor) * factor, (ny // factor) * factor, (nx // factor) * factor
    cropped = xp.asarray(volume[:cz, :cy, :cx], dtype=xp.float32)
    return cropped.reshape(
        cz // factor, factor, cy // factor, factor, cx // factor, factor
    ).mean(axis=(1, 3, 5))


def _pyramid_factors(shape: tuple[int, int, int]) -> list[int]:
    """Pyramid factors (coarsest first) that keep every level usably large."""
    smallest = min(shape)
    factors = [f for f in _PYRAMID_FACTORS if f == 1 or smallest // f >= _MIN_LEVEL_SIZE]
    return factors or [1]


def _sample_positions(shape: tuple[int, int, int], margins: tuple[int, int, int], xp=np):
    """Strided grid of interior sample positions, as flat indices.

    Positions stay at least *margins* voxels from every border, so any
    candidate shift within the margins can be gathered without bounds checks.
    """
    nx = shape[2]
    ny = shape[1]
    axes = []
    for n, m in zip(shape, margins):
        axes.append((m, n - m))

    counts = [max(1, hi - lo) for lo, hi in axes]
    total = counts[0] * counts[1] * counts[2]
    stride = max(1, math.ceil((total / _MAX_SAMPLES) ** (1.0 / 3.0)))

    zs = xp.arange(axes[0][0], axes[0][1], stride)
    ys = xp.arange(axes[1][0], axes[1][1], stride)
    xs = xp.arange(axes[2][0], axes[2][1], stride)

    flat = (
        (zs[:, None, None] * ny + ys[None, :, None]) * nx + xs[None, None, :]
    ).ravel()
    return flat.astype(xp.int64)


def _evaluate_costs(
    desc_ref,
    desc_mov,
    shifts: np.ndarray,
    base_indices,
    xp,
    progress_callback: ProgressCallback | None,
    f0: float,
    f1: float,
):
    """Mean descriptor SSD for every candidate shift, evaluated in batches.

    ``cost[i]`` compares ``desc_ref`` at ``p + shifts[i]`` with ``desc_mov`` at
    ``p``, averaged over the sample positions ``p`` — so the returned best shift
    is the correction to apply to the moving volume, matching the convention of
    the other algorithms.
    """
    n_shifts = shifts.shape[0]
    if n_shifts == 0:
        return np.empty(0, dtype=np.float64)

    ny, nx = desc_ref.shape[2:]
    flat_ref = desc_ref.reshape(_N_DESCRIPTORS, -1)
    mov_samples = desc_mov.reshape(_N_DESCRIPTORS, -1)[:, base_indices]

    n_samples = int(base_indices.shape[0])
    offsets_np = shifts[:, 0] * (ny * nx) + shifts[:, 1] * nx + shifts[:, 2]
    offsets = xp.asarray(offsets_np, dtype=xp.int64)

    batch = max(1, _BATCH_ELEMENTS // max(1, _N_DESCRIPTORS * n_samples))
    # float64 accumulation: candidate costs differ in the last digits of a
    # float32 mean over ~10^5 terms, and that difference is what ranks them.
    costs = xp.empty(n_shifts, dtype=xp.float64)

    for start in range(0, n_shifts, batch):
        stop = min(start + batch, n_shifts)
        idx = base_indices[None, :] + offsets[start:stop, None]
        gathered = flat_ref[:, idx.ravel()].reshape(
            _N_DESCRIPTORS, stop - start, n_samples
        )
        gathered -= mov_samples[:, None, :]
        gathered *= gathered
        costs[start:stop] = gathered.mean(axis=(0, 2), dtype=xp.float64)
        _report(progress_callback, f0 + (f1 - f0) * (stop / n_shifts))

    return np.asarray(costs.get() if hasattr(costs, "get") else costs, dtype=np.float64)


def _build_shifts(
    center: tuple[int, int, int],
    radius: int,
    limits: tuple[int, int, int],
) -> np.ndarray:
    """Candidate ``(dz, dy, dx)`` shifts around *center*, clipped to *limits*."""
    cz, cy, cx = center
    lz, ly, lx = limits
    zs = [d for d in range(cz - radius, cz + radius + 1) if abs(d) <= lz]
    ys = [d for d in range(cy - radius, cy + radius + 1) if abs(d) <= ly]
    xs = [d for d in range(cx - radius, cx + radius + 1) if abs(d) <= lx]
    if not (zs and ys and xs):
        return np.zeros((1, 3), dtype=np.int64)
    return np.array(
        [(dz, dy, dx) for dz in zs for dy in ys for dx in xs], dtype=np.int64
    )


class DeedsRegistration(RegistrationAlgorithm):
    """Translation search on MIND-SSC descriptors, deedsBCV-style.

    Parameters
    ----------
    quantisation_step : int
        Self-similarity spacing ``qs`` of the descriptor, in voxels of the
        level being processed (deedsBCV's default is 1).
    refine_radius : int
        Half-width, in voxels of the current level, of the neighbourhood
        re-searched after each pyramid step.
    """

    def __init__(self, quantisation_step: int = 1, refine_radius: int = 3) -> None:
        self.quantisation_step = int(quantisation_step)
        self.refine_radius = int(refine_radius)

    def register(
        self,
        reference_volume: np.ndarray,
        moving_volume: np.ndarray,
        search_range_xy: int,
        search_range_z: int,
        use_gpu: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> RegistrationResult:
        if use_gpu:
            try:
                import cupy as cp

                return self._register_with(
                    cp, reference_volume, moving_volume,
                    search_range_xy, search_range_z, progress_callback,
                )
            except Exception as exc:
                logger.warning(
                    "GPU deedsBCV registration failed (%s), falling back to CPU", exc
                )

        return self._register_with(
            np, reference_volume, moving_volume,
            search_range_xy, search_range_z, progress_callback,
        )

    # ------------------------------------------------------------------ #
    # Shared CPU/GPU implementation (xp is numpy or cupy)
    # ------------------------------------------------------------------ #

    def _register_with(
        self,
        xp,
        ref: np.ndarray,
        mov: np.ndarray,
        sr_xy: int,
        sr_z: int,
        progress_callback: ProgressCallback | None = None,
    ) -> RegistrationResult:
        factors = _pyramid_factors(ref.shape)
        n_levels = len(factors)

        best_shift = (0, 0, 0)  # in full-resolution voxels
        coarse_costs: np.ndarray | None = None
        best_cost = 0.0

        for level, factor in enumerate(factors):
            f0 = level / n_levels
            f1 = (level + 1) / n_levels

            ref_level = _downsample(ref, factor, xp)
            mov_level = _downsample(mov, factor, xp)
            desc_ref = mind_ssc(ref_level, self.quantisation_step, xp)
            desc_mov = mind_ssc(mov_level, self.quantisation_step, xp)

            shape = tuple(int(s) for s in ref_level.shape)
            # A shift can only be tested if sample positions that far from the
            # border exist, which caps the range on thin axes.
            limits = (
                min(sr_z // factor, (shape[0] - 1) // 2),
                min(sr_xy // factor, (shape[1] - 1) // 2),
                min(sr_xy // factor, (shape[2] - 1) // 2),
            )
            base_indices = _sample_positions(shape, limits, xp)

            if level == 0:
                center = (0, 0, 0)
                radius = max(limits)
            else:
                previous = factors[level - 1]
                center = (
                    int(round(best_shift[0] / factor)),
                    int(round(best_shift[1] / factor)),
                    int(round(best_shift[2] / factor)),
                )
                radius = max(self.refine_radius, previous // factor + 1)

            shifts = _build_shifts(center, radius, limits)
            costs = _evaluate_costs(
                desc_ref, desc_mov, shifts, base_indices, xp,
                progress_callback, f0, f1,
            )

            best_idx = int(np.argmin(costs))
            best_cost = float(costs[best_idx])
            best_shift = tuple(int(v) * factor for v in shifts[best_idx])

            if coarse_costs is None:
                coarse_costs = costs

            del desc_ref, desc_mov, ref_level, mov_level
            if xp is not np:
                xp.get_default_memory_pool().free_all_blocks()

        _report(progress_callback, 1.0)

        confidence = self._compute_confidence(
            coarse_costs if coarse_costs is not None else np.empty(0)
        )

        return RegistrationResult(
            shift_x=best_shift[2],
            shift_y=best_shift[1],
            shift_z=best_shift[0],
            confidence=confidence,
            raw_metric_value=best_cost,
            algorithm_name=ALGORITHM_NAME,
        )

    @staticmethod
    def _compute_confidence(costs: np.ndarray) -> float:
        """How far the best candidate stands out from the cost distribution.

        Mirrors the mutual-information confidence, with the sign flipped
        because descriptor SSD is minimised: ``(median - min) / (max - min)``.
        The coarsest level is used, since it is the only pass that sees the
        whole search range.
        """
        if costs.size < 2:
            return 0.0
        c_min = float(costs.min())
        c_max = float(costs.max())
        denom = c_max - c_min
        if denom < 1e-12:
            return 0.0
        return max(0.0, min(1.0, (float(np.median(costs)) - c_min) / denom))
