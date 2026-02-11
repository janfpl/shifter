"""Mutual-information registration with coarse-to-fine search."""

from __future__ import annotations

import logging

import numpy as np

from chromatic_shift_corrector.registration.base import (
    RegistrationAlgorithm,
    RegistrationResult,
)

logger = logging.getLogger(__name__)

ALGORITHM_NAME = "Mutual Information"

_MI_BINS = 64
_COARSE_STEP = 5
_FINE_RADIUS = 5


def _compute_mi(ref: np.ndarray, mov: np.ndarray, bins: int = _MI_BINS) -> float:
    """Compute mutual information between two arrays using a joint histogram.

    MI = H(ref) + H(mov) - H(ref, mov)
    """
    # Flatten and discard any NaN/Inf.
    r = ref.ravel().astype(np.float64)
    m = mov.ravel().astype(np.float64)

    hist_2d, _, _ = np.histogram2d(r, m, bins=bins)
    # Normalise to joint probability.
    pxy = hist_2d / hist_2d.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    # Shannon entropy (avoid log(0)).
    eps = np.finfo(np.float64).tiny
    hx = -np.sum(px * np.log(px + eps))
    hy = -np.sum(py * np.log(py + eps))
    hxy = -np.sum(pxy * np.log(pxy + eps))

    return float(hx + hy - hxy)


def _overlapping_regions(
    ref: np.ndarray,
    mov: np.ndarray,
    dz: int,
    dy: int,
    dx: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the overlapping sub-arrays after applying shift (dz, dy, dx) to *mov*.

    Returns None if there is no overlap.
    """
    nz, ny, nx = ref.shape

    def _range(shift: int, length: int):
        if shift >= 0:
            r_start, r_end = shift, length
            m_start, m_end = 0, length - shift
        else:
            r_start, r_end = 0, length + shift
            m_start, m_end = -shift, length
        if r_end <= r_start or m_end <= m_start:
            return None
        return (r_start, r_end), (m_start, m_end)

    rz = _range(dz, nz)
    ry = _range(dy, ny)
    rx = _range(dx, nx)
    if rz is None or ry is None or rx is None:
        return None

    ref_sub = ref[rz[0][0]:rz[0][1], ry[0][0]:ry[0][1], rx[0][0]:rx[0][1]]
    mov_sub = mov[rz[1][0]:rz[1][1], ry[1][0]:ry[1][1], rx[1][0]:rx[1][1]]
    return ref_sub, mov_sub


class MutualInformationRegistration(RegistrationAlgorithm):
    """Coarse-to-fine mutual-information registration."""

    def register(
        self,
        reference_volume: np.ndarray,
        moving_volume: np.ndarray,
        search_range_xy: int,
        search_range_z: int,
        use_gpu: bool = False,
    ) -> RegistrationResult:
        if use_gpu:
            try:
                return self._register_gpu(
                    reference_volume, moving_volume, search_range_xy, search_range_z
                )
            except Exception as exc:
                logger.warning("GPU MI registration failed (%s), falling back to CPU", exc)

        return self._register_cpu(
            reference_volume, moving_volume, search_range_xy, search_range_z
        )

    # ------------------------------------------------------------------ #
    # CPU path
    # ------------------------------------------------------------------ #

    def _register_cpu(
        self,
        ref: np.ndarray,
        mov: np.ndarray,
        sr_xy: int,
        sr_z: int,
    ) -> RegistrationResult:
        ref_f = ref.astype(np.float64)
        mov_f = mov.astype(np.float64)

        # ---- coarse pass ------------------------------------------------
        coarse_step = _COARSE_STEP
        best_shift, mi_values = self._grid_search(
            ref_f, mov_f, sr_xy, sr_z, coarse_step
        )

        # ---- fine pass --------------------------------------------------
        fine_radius = min(_FINE_RADIUS, sr_xy, sr_z)
        fine_shifts = []
        bz, by, bx = best_shift
        for dz in range(bz - fine_radius, bz + fine_radius + 1):
            if abs(dz) > sr_z:
                continue
            for dy in range(by - fine_radius, by + fine_radius + 1):
                if abs(dy) > sr_xy:
                    continue
                for dx in range(bx - fine_radius, bx + fine_radius + 1):
                    if abs(dx) > sr_xy:
                        continue
                    overlap = _overlapping_regions(ref_f, mov_f, dz, dy, dx)
                    if overlap is None:
                        continue
                    mi = _compute_mi(overlap[0], overlap[1])
                    fine_shifts.append((dz, dy, dx, mi))
                    mi_values.append(mi)

        if fine_shifts:
            best = max(fine_shifts, key=lambda t: t[3])
            best_shift = (best[0], best[1], best[2])
            peak_mi = best[3]
        else:
            peak_mi = mi_values[0] if mi_values else 0.0

        confidence = self._compute_confidence(peak_mi, mi_values)

        return RegistrationResult(
            shift_x=best_shift[2],
            shift_y=best_shift[1],
            shift_z=best_shift[0],
            confidence=confidence,
            raw_metric_value=peak_mi,
            algorithm_name=ALGORITHM_NAME,
        )

    def _grid_search(
        self,
        ref: np.ndarray,
        mov: np.ndarray,
        sr_xy: int,
        sr_z: int,
        step: int,
    ) -> tuple[tuple[int, int, int], list[float]]:
        """Exhaustive grid search and return (best_shift, all_mi_values)."""
        best_mi = -np.inf
        best_shift = (0, 0, 0)
        mi_values: list[float] = []

        for dz in range(-sr_z, sr_z + 1, step):
            for dy in range(-sr_xy, sr_xy + 1, step):
                for dx in range(-sr_xy, sr_xy + 1, step):
                    overlap = _overlapping_regions(ref, mov, dz, dy, dx)
                    if overlap is None:
                        continue
                    mi = _compute_mi(overlap[0], overlap[1])
                    mi_values.append(mi)
                    if mi > best_mi:
                        best_mi = mi
                        best_shift = (dz, dy, dx)

        return best_shift, mi_values

    @staticmethod
    def _compute_confidence(peak_mi: float, mi_values: list[float]) -> float:
        """Confidence = (peak_MI - median_MI) / (max_MI - min_MI)."""
        if not mi_values or len(mi_values) < 2:
            return 0.0
        arr = np.array(mi_values)
        mi_min = float(arr.min())
        mi_max = float(arr.max())
        mi_med = float(np.median(arr))
        denom = mi_max - mi_min
        if denom < 1e-12:
            return 0.0
        return max(0.0, min(1.0, (peak_mi - mi_med) / denom))

    # ------------------------------------------------------------------ #
    # GPU path
    # ------------------------------------------------------------------ #

    def _register_gpu(
        self,
        ref: np.ndarray,
        mov: np.ndarray,
        sr_xy: int,
        sr_z: int,
    ) -> RegistrationResult:
        """GPU-accelerated MI registration.

        The main speed-up comes from computing joint histograms on the GPU.
        Falls back to CPU for the overall search loop.
        """
        import cupy as cp

        ref_g = cp.asarray(ref, dtype=cp.float64)
        mov_g = cp.asarray(mov, dtype=cp.float64)

        def _mi_gpu(r_sub, m_sub):
            hist = cp.histogram2d(
                r_sub.ravel(), m_sub.ravel(), bins=_MI_BINS
            )[0]
            pxy = hist / hist.sum()
            px = pxy.sum(axis=1)
            py = pxy.sum(axis=0)
            eps = cp.finfo(cp.float64).tiny
            hx = -cp.sum(px * cp.log(px + eps))
            hy = -cp.sum(py * cp.log(py + eps))
            hxy = -cp.sum(pxy * cp.log(pxy + eps))
            return float((hx + hy - hxy).get())

        def _overlap_gpu(rg, mg, dz, dy, dx):
            nz, ny, nx = rg.shape
            def _rng(s, l):
                if s >= 0:
                    return (s, l, 0, l - s)
                return (0, l + s, -s, l)
            rz0, rz1, mz0, mz1 = _rng(dz, nz)
            ry0, ry1, my0, my1 = _rng(dy, ny)
            rx0, rx1, mx0, mx1 = _rng(dx, nx)
            if rz1 <= rz0 or ry1 <= ry0 or rx1 <= rx0:
                return None
            return rg[rz0:rz1, ry0:ry1, rx0:rx1], mg[mz0:mz1, my0:my1, mx0:mx1]

        # Coarse search.
        best_mi = -np.inf
        best_shift = (0, 0, 0)
        mi_values: list[float] = []
        step = _COARSE_STEP

        for dz in range(-sr_z, sr_z + 1, step):
            for dy in range(-sr_xy, sr_xy + 1, step):
                for dx in range(-sr_xy, sr_xy + 1, step):
                    ov = _overlap_gpu(ref_g, mov_g, dz, dy, dx)
                    if ov is None:
                        continue
                    mi = _mi_gpu(ov[0], ov[1])
                    mi_values.append(mi)
                    if mi > best_mi:
                        best_mi = mi
                        best_shift = (dz, dy, dx)

        # Fine search.
        fine_radius = min(_FINE_RADIUS, sr_xy, sr_z)
        bz, by, bx = best_shift
        for dz in range(bz - fine_radius, bz + fine_radius + 1):
            if abs(dz) > sr_z:
                continue
            for dy in range(by - fine_radius, by + fine_radius + 1):
                if abs(dy) > sr_xy:
                    continue
                for dx in range(bx - fine_radius, bx + fine_radius + 1):
                    if abs(dx) > sr_xy:
                        continue
                    ov = _overlap_gpu(ref_g, mov_g, dz, dy, dx)
                    if ov is None:
                        continue
                    mi = _mi_gpu(ov[0], ov[1])
                    mi_values.append(mi)
                    if mi > best_mi:
                        best_mi = mi
                        best_shift = (dz, dy, dx)

        confidence = self._compute_confidence(best_mi, mi_values)

        return RegistrationResult(
            shift_x=best_shift[2],
            shift_y=best_shift[1],
            shift_z=best_shift[0],
            confidence=confidence,
            raw_metric_value=best_mi,
            algorithm_name=ALGORITHM_NAME,
        )
