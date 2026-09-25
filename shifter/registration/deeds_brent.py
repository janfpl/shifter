"""deedsBCV (MIND-SSC) registration refined with Brent's method.

A variant of :mod:`shifter.registration.deeds` that keeps the MIND-SSC
self-similarity descriptor but replaces the discrete coarse-to-fine grid search
with **Brent's method** — the bounded one-dimensional optimizer from
:func:`scipy.optimize.minimize_scalar` (``method="bounded"``) — applied per axis
in a cyclic coordinate-descent loop, mirroring the Mutual Information (Brent)
method.

How it works
------------
1. **Coarse seed.** Compute MIND-SSC descriptors on the *coarsest* pyramid level
   (a downsampled volume) and grid-search the whole range there for a starting
   shift. This is deeds' first pyramid level, and it locates the correct basin
   cheaply — the descriptor cost surface is multimodal over a translation, so a
   purely local optimizer started from zero could get trapped.
2. **Brent refinement.** Compute descriptors once at full resolution, then
   refine the seed with Brent per axis. The descriptor cost is evaluated at
   *continuous* shifts by linear interpolation of the fixed descriptor field
   (:func:`scipy.ndimage.map_coordinates`), so Brent sees a smooth objective
   rather than the step function an integer-only gather would give. It replaces
   the finer pyramid levels' grid searches with far fewer evaluations.

The converged continuous shift is rounded to the nearest voxel for the result,
matching the application's integer-shift model.

Like the Mutual Information (Brent) method, this runs on CPU: Brent is a
sequential optimizer, so ``use_gpu`` is accepted but not used.

Scope note (inherited from deeds): this is deedsBCV's similarity core in a
translation-only role, not the full deformable registration — see
:mod:`shifter.registration.deeds`.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy import ndimage as _ndi
from scipy.optimize import minimize_scalar

from shifter.registration.base import (
    ProgressCallback,
    RegistrationAlgorithm,
    RegistrationResult,
)
from shifter.registration.timing import note, phase
from shifter.registration.deeds import (
    DeedsRegistration,
    _build_shifts,
    _downsample,
    _evaluate_costs,
    _N_DESCRIPTORS,
    _pyramid_factors,
    _report,
    _sample_positions,
    mind_ssc,
)

logger = logging.getLogger(__name__)

ALGORITHM_NAME = "deedsBCV (MIND-SSC, Brent)"

# Cap on sample points used per Brent objective evaluation. Smaller than the
# grid method's budget because Brent performs many sequential evaluations.
_MAX_BRENT_SAMPLES = 8000

# Coordinate-descent controls (shared spirit with the MI (Brent) method).
_MAX_SWEEPS = 3
_BRENT_XATOL = 0.1
_BRENT_MAXITER = 40
_CONVERGE_TOL = 0.1

_INVALID = 1e12


def _sample_base_coords(
    shape: tuple[int, int, int],
    seed: tuple[int, int, int],
    margin: int,
) -> np.ndarray | None:
    """Strided interior base coords ``p`` (as a ``(3, K)`` float array).

    Points are chosen so that both ``p`` (where the moving descriptor is
    sampled) and ``p + seed`` (where the reference descriptor is interpolated)
    stay at least *margin* voxels from every border — leaving room for Brent to
    nudge the reference sample by up to *margin* without going out of bounds.
    Returns ``None`` if no such interior region exists.
    """
    ranges = []
    for axis in range(3):
        lo = max(margin, margin - seed[axis])
        hi = min(shape[axis] - margin, shape[axis] - margin - seed[axis])
        if hi <= lo:
            return None
        ranges.append((lo, hi))

    total = 1
    for lo, hi in ranges:
        total *= hi - lo
    stride = max(1, math.ceil((total / _MAX_BRENT_SAMPLES) ** (1.0 / 3.0)))

    zs = np.arange(ranges[0][0], ranges[0][1], stride)
    ys = np.arange(ranges[1][0], ranges[1][1], stride)
    xs = np.arange(ranges[2][0], ranges[2][1], stride)
    if zs.size == 0 or ys.size == 0 or xs.size == 0:
        return None

    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
    return np.stack([zz.ravel(), yy.ravel(), xx.ravel()]).astype(np.float64)


class DeedsBrentRegistration(RegistrationAlgorithm):
    """MIND-SSC registration refined by Brent's method.

    Parameters
    ----------
    quantisation_step : int
        Self-similarity spacing ``qs`` of the MIND-SSC descriptor (deeds'
        default is 1).
    """

    def __init__(self, quantisation_step: int = 1) -> None:
        self.quantisation_step = int(quantisation_step)

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
            logger.debug(
                "%s runs on CPU (Brent is a sequential optimizer); ignoring use_gpu",
                ALGORITHM_NAME,
            )

        qs = self.quantisation_step
        ref = np.ascontiguousarray(reference_volume, dtype=np.float32)
        mov = np.ascontiguousarray(moving_volume, dtype=np.float32)
        sr_xy, sr_z = search_range_xy, search_range_z
        limits_full = (sr_z, sr_xy, sr_xy)

        # ---- coarse seed on the coarsest pyramid level -----------------
        with phase(f"{ALGORITHM_NAME} coarse seed"):
            seed, coarse_costs, factor = self._coarse_seed(
                ref, mov, sr_xy, sr_z, qs, progress_callback
            )

        # ---- full-resolution descriptors -------------------------------
        with phase(f"{ALGORITHM_NAME} full-res descriptors"):
            desc_ref = mind_ssc(ref, qs, np)
            desc_mov = mind_ssc(mov, qs, np)
        shape = tuple(int(s) for s in ref.shape)

        # Brent may nudge the reference sample by up to this many voxels.
        radius = factor + 2
        base = _sample_base_coords(shape, seed, margin=radius + 1)

        if base is None:
            # No interior region (tiny volume / large seed) — keep the seed.
            _report(progress_callback, 1.0)
            best_shift = tuple(
                int(np.clip(seed[i], -limits_full[i], limits_full[i])) for i in range(3)
            )
            confidence = DeedsRegistration._compute_confidence(coarse_costs)
            return self._result(best_shift, 0.0, confidence)

        bz, by, bx = (base[0].astype(np.int64), base[1].astype(np.int64), base[2].astype(np.int64))
        mov_samples = np.stack([desc_mov[c][bz, by, bx] for c in range(_N_DESCRIPTORS)])
        seed_col = np.asarray(seed, dtype=np.float64)[:, None]

        n_evals = 0

        def cost(delta: tuple[float, float, float]) -> float:
            nonlocal n_evals
            n_evals += 1
            coords = base + seed_col + np.asarray(delta, dtype=np.float64)[:, None]
            total = 0.0
            for c in range(_N_DESCRIPTORS):
                gathered = _ndi.map_coordinates(
                    desc_ref[c], coords, order=1, mode="nearest"
                )
                diff = gathered - mov_samples[c]
                total += float(np.dot(diff, diff))
            return total / (_N_DESCRIPTORS * base.shape[1])

        # ---- Brent refinement: cyclic per-axis line search -------------
        cur = [0.0, 0.0, 0.0]  # residual delta around the seed
        with phase(f"{ALGORITHM_NAME} Brent refine"):
            for sweep in range(_MAX_SWEEPS):
                prev = list(cur)
                for axis in range(3):
                    # Bound the residual so the total shift (seed + delta) stays
                    # in the search range, and within the interpolation radius.
                    lo = max(-radius, -limits_full[axis] - seed[axis])
                    hi = min(radius, limits_full[axis] - seed[axis])
                    if hi - lo < 1e-3:
                        continue

                    def objective(x: float, axis: int = axis) -> float:
                        trial = list(cur)
                        trial[axis] = x
                        return cost(trial)

                    res = minimize_scalar(
                        objective,
                        bounds=(lo, hi),
                        method="bounded",
                        options={"xatol": _BRENT_XATOL, "maxiter": _BRENT_MAXITER},
                    )
                    if np.isfinite(res.fun):
                        cur[axis] = float(res.x)

                _report(progress_callback, 0.4 + 0.6 * (sweep + 1) / _MAX_SWEEPS)
                if max(abs(cur[i] - prev[i]) for i in range(3)) < _CONVERGE_TOL:
                    break

        note(f"{ALGORITHM_NAME}: {n_evals} descriptor-cost evaluations")
        _report(progress_callback, 1.0)

        best_cost = cost(cur)
        best_shift = tuple(
            int(np.clip(round(seed[i] + cur[i]), -limits_full[i], limits_full[i]))
            for i in range(3)
        )
        confidence = DeedsRegistration._compute_confidence(coarse_costs)
        return self._result(best_shift, best_cost, confidence)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _coarse_seed(
        self,
        ref: np.ndarray,
        mov: np.ndarray,
        sr_xy: int,
        sr_z: int,
        qs: int,
        progress_callback: ProgressCallback | None,
    ) -> tuple[tuple[int, int, int], np.ndarray, int]:
        """Grid-search the coarsest pyramid level for a seed shift.

        Returns ``(seed_full_res, coarse_costs, factor)``. Reports progress
        across ``[0.0, 0.4]``.
        """
        factor = _pyramid_factors(ref.shape)[0]
        ref_c = _downsample(ref, factor, np)
        mov_c = _downsample(mov, factor, np)
        desc_ref_c = mind_ssc(ref_c, qs, np)
        desc_mov_c = mind_ssc(mov_c, qs, np)

        shape_c = tuple(int(s) for s in ref_c.shape)
        limits_c = (
            min(sr_z // factor, (shape_c[0] - 1) // 2),
            min(sr_xy // factor, (shape_c[1] - 1) // 2),
            min(sr_xy // factor, (shape_c[2] - 1) // 2),
        )
        base_c = _sample_positions(shape_c, limits_c, np)
        shifts_c = _build_shifts((0, 0, 0), max(limits_c), limits_c)
        costs_c = _evaluate_costs(
            desc_ref_c, desc_mov_c, shifts_c, base_c, np, progress_callback, 0.0, 0.4
        )

        best_idx = int(np.argmin(costs_c))
        seed = tuple(int(shifts_c[best_idx, k]) * factor for k in range(3))
        return seed, costs_c, factor

    @staticmethod
    def _result(
        best_shift: tuple[int, int, int], raw: float, confidence: float
    ) -> RegistrationResult:
        return RegistrationResult(
            shift_x=best_shift[2],
            shift_y=best_shift[1],
            shift_z=best_shift[0],
            confidence=confidence,
            raw_metric_value=raw,
            algorithm_name=ALGORITHM_NAME,
        )
