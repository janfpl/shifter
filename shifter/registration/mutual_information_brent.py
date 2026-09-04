"""Mutual-information registration refined with Brent's method.

A variant of :mod:`shifter.registration.mutual_information` that keeps the same
similarity metric (mutual information of a joint intensity histogram) but
replaces the exhaustive *fine* grid search with **Brent's method** — the bounded
one-dimensional optimizer behind :func:`scipy.optimize.minimize_scalar`
(``method="bounded"``, i.e. ``scipy.optimize.fminbound``). It is applied per axis
in a cyclic coordinate-descent loop.

Why a coarse grid *and* Brent
-----------------------------
Mutual information over a translation is strongly multimodal — many local
maxima across the search box — so a purely local optimizer started from zero
would routinely settle into the wrong basin. This method therefore keeps a cheap
coarse grid pass (step :data:`_COARSE_STEP`, reused from the grid MI algorithm)
to locate the right basin, then hands off to Brent for the refinement the grid
algorithm would otherwise do with an exhaustive ``±_FINE_RADIUS`` scan. Brent
reaches the optimum with far fewer MI evaluations than that fine grid.

Sub-voxel evaluation, integer result
-------------------------------------
Brent optimizes a *continuous* shift. The integer part is applied by the same
exact overlap slicing the grid MI uses (no interpolation, any magnitude); the
sub-voxel remainder (|frac| <= 0.5) is applied to the moving sub-volume with
linear interpolation, so the objective Brent sees is smooth rather than a step
function. The converged continuous shift is rounded to the nearest voxel for the
result, matching the application's integer-shift model.

Runs on CPU: Brent is an inherently sequential optimizer, and offloading each
small per-evaluation histogram to the GPU would cost more in transfers than it
saves, so ``use_gpu`` is accepted but not used here.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import ndimage as _ndi
from scipy.optimize import minimize_scalar

from shifter.registration.base import (
    ProgressCallback,
    RegistrationAlgorithm,
    RegistrationResult,
)
from shifter.registration.timing import note, phase
from shifter.registration.mutual_information import (
    _MI_BINS,
    _COARSE_STEP,
    _build_shifts_array,
    _compute_mi,
    _HAVE_NUMBA,
    _overlapping_regions,
    _report,
    MutualInformationRegistration,
)

logger = logging.getLogger(__name__)

ALGORITHM_NAME = "Mutual Information (Brent)"

# Half-width (voxels) of the per-axis Brent bracket around the current estimate.
# Matches the coarse grid step so Brent can reach the true optimum lying anywhere
# between two coarse nodes, and re-centres each sweep so it can migrate further.
_BRENT_RADIUS = _COARSE_STEP

# Coordinate-descent controls.
_MAX_SWEEPS = 3          # cycles over (Z, Y, X)
_BRENT_XATOL = 0.1       # voxels — Brent stops refining an axis below this
_BRENT_MAXITER = 40
_CONVERGE_TOL = 0.1      # voxels — stop sweeping once the estimate barely moves

# Penalty returned for shifts with too little overlap (kept finite so Brent's
# parabolic step is well-behaved).
_INVALID = 1e6


def _neg_mi_continuous(
    ref: np.ndarray,
    mov: np.ndarray,
    sz: float,
    sy: float,
    sx: float,
    bins: int = _MI_BINS,
) -> float:
    """Return ``-MI`` for a continuous shift ``(sz, sy, sx)``.

    The integer part uses exact overlap slicing (as in the grid MI); the
    fractional remainder is applied by linear interpolation. A one-voxel border
    is always trimmed so every evaluation compares the same-sized region,
    keeping the objective continuous as ``frac`` crosses an integer.
    """
    nz, ny, nx = int(round(sz)), int(round(sy)), int(round(sx))
    fz, fy, fx = sz - nz, sy - ny, sx - nx

    overlap = _overlapping_regions(ref, mov, nz, ny, nx)
    if overlap is None:
        return _INVALID
    ref_sub, mov_sub = overlap
    if any(d < 3 for d in ref_sub.shape):
        return _INVALID

    if abs(fz) > 1e-3 or abs(fy) > 1e-3 or abs(fx) > 1e-3:
        mov_sub = _ndi.shift(mov_sub, (fz, fy, fx), order=1, mode="nearest")

    ref_sub = ref_sub[1:-1, 1:-1, 1:-1]
    mov_sub = mov_sub[1:-1, 1:-1, 1:-1]
    return -_compute_mi(ref_sub, mov_sub, bins)


class MutualInformationBrentRegistration(RegistrationAlgorithm):
    """Mutual-information registration refined by Brent's method."""

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

        ref = np.ascontiguousarray(reference_volume, dtype=np.float64)
        mov = np.ascontiguousarray(moving_volume, dtype=np.float64)
        sr_xy, sr_z = search_range_xy, search_range_z

        # ---- coarse grid pass: locate the basin ------------------------
        with phase(f"{ALGORITHM_NAME} coarse seed"):
            seed, mi_values = self._coarse_seed(
                ref, mov, sr_xy, sr_z, progress_callback
            )

        # ---- Brent refinement: cyclic per-axis line search -------------
        # Each objective evaluation computes a full-overlap mutual-information
        # histogram (single-threaded numpy) plus a sub-voxel interpolation, so on
        # a large ROI this refinement — not the (numba-parallel) coarse seed —
        # dominates. The eval count and timing are logged to make that visible.
        cur = [float(seed[0]), float(seed[1]), float(seed[2])]
        limits = (sr_z, sr_xy, sr_xy)
        n_evals = 0

        with phase(f"{ALGORITHM_NAME} Brent refine"):
            for sweep in range(_MAX_SWEEPS):
                prev = list(cur)
                for axis in range(3):
                    lo = max(-limits[axis], cur[axis] - _BRENT_RADIUS)
                    hi = min(limits[axis], cur[axis] + _BRENT_RADIUS)
                    if hi - lo < 1e-3:
                        continue

                    def objective(x: float, axis: int = axis) -> float:
                        nonlocal n_evals
                        n_evals += 1
                        trial = list(cur)
                        trial[axis] = x
                        val = _neg_mi_continuous(ref, mov, trial[0], trial[1], trial[2])
                        if val < _INVALID:
                            mi_values.append(-val)
                        return val

                    res = minimize_scalar(
                        objective,
                        bounds=(lo, hi),
                        method="bounded",
                        options={"xatol": _BRENT_XATOL, "maxiter": _BRENT_MAXITER},
                    )
                    if np.isfinite(res.fun) and res.fun < _INVALID:
                        cur[axis] = float(res.x)

                _report(
                    progress_callback,
                    0.5 + 0.5 * (sweep + 1) / _MAX_SWEEPS,
                )
                if max(abs(cur[i] - prev[i]) for i in range(3)) < _CONVERGE_TOL:
                    break

        note(f"{ALGORITHM_NAME}: {n_evals} MI objective evaluations")
        _report(progress_callback, 1.0)

        # ---- round to the integer-shift model --------------------------
        best_shift = tuple(
            int(np.clip(round(cur[i]), -limits[i], limits[i])) for i in range(3)
        )
        peak_mi = -_neg_mi_continuous(ref, mov, *(float(v) for v in best_shift))
        if peak_mi < 0:  # invalid overlap at the rounded point (shouldn't happen)
            peak_mi = max(mi_values) if mi_values else 0.0

        confidence = MutualInformationRegistration._compute_confidence(
            peak_mi, mi_values
        )

        return RegistrationResult(
            shift_x=best_shift[2],
            shift_y=best_shift[1],
            shift_z=best_shift[0],
            confidence=confidence,
            raw_metric_value=peak_mi,
            algorithm_name=ALGORITHM_NAME,
        )

    # ------------------------------------------------------------------ #
    # Coarse seed
    # ------------------------------------------------------------------ #

    def _coarse_seed(
        self,
        ref: np.ndarray,
        mov: np.ndarray,
        sr_xy: int,
        sr_z: int,
        progress_callback: ProgressCallback | None,
    ) -> tuple[tuple[int, int, int], list[float]]:
        """Coarse grid search (step ``_COARSE_STEP``) for a starting shift.

        Uses the numba-parallel grid from the grid MI module when available,
        otherwise a serial numpy scan. Returns ``(best_shift, mi_values)`` and
        reports progress across ``[0.0, 0.5]``.
        """
        if _HAVE_NUMBA:
            from shifter.registration.mutual_information import _grid_search_batched

            shifts = _build_shifts_array(sr_xy, sr_z, _COARSE_STEP)
            mi = _grid_search_batched(
                ref, mov, shifts, _MI_BINS, progress_callback, 0.0, 0.5
            )
            best_idx = int(np.argmax(mi))
            best = (
                int(shifts[best_idx, 0]),
                int(shifts[best_idx, 1]),
                int(shifts[best_idx, 2]),
            )
            return best, list(mi)

        # Serial fallback.
        best_mi = -np.inf
        best = (0, 0, 0)
        mi_values: list[float] = []
        zs = range(-sr_z, sr_z + 1, _COARSE_STEP)
        ys = range(-sr_xy, sr_xy + 1, _COARSE_STEP)
        xs = range(-sr_xy, sr_xy + 1, _COARSE_STEP)
        total = max(1, len(zs) * len(ys) * len(xs))
        seen = 0
        for dz in zs:
            for dy in ys:
                for dx in xs:
                    seen += 1
                    if seen % 32 == 0:
                        _report(progress_callback, 0.5 * seen / total)
                    overlap = _overlapping_regions(ref, mov, dz, dy, dx)
                    if overlap is None:
                        continue
                    mi = _compute_mi(overlap[0], overlap[1])
                    mi_values.append(mi)
                    if mi > best_mi:
                        best_mi = mi
                        best = (dz, dy, dx)
        _report(progress_callback, 0.5)
        return best, mi_values
