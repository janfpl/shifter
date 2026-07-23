"""Mutual-information registration with coarse-to-fine search.

CPU path uses numba-parallelised grid search when numba is available,
falling back to a pure-numpy implementation otherwise.
"""

from __future__ import annotations

import logging

import numpy as np

from shifter.registration.base import (
    RegistrationAlgorithm,
    RegistrationResult,
)

logger = logging.getLogger(__name__)

ALGORITHM_NAME = "Mutual Information"

_MI_BINS = 64
_COARSE_STEP = 5
_FINE_RADIUS = 5

# ---------------------------------------------------------------------------
# Try to import numba; fall back gracefully if unavailable.
# ---------------------------------------------------------------------------
try:
    import numba

    @numba.njit(cache=True)
    def _joint_histogram(r_flat: np.ndarray, m_flat: np.ndarray, bins: int) -> np.ndarray:
        """Build a joint histogram from two flattened float64 arrays."""
        hist = np.zeros((bins, bins), dtype=np.float64)
        n = r_flat.shape[0]
        if n == 0:
            return hist

        r_min = r_flat[0]
        r_max = r_flat[0]
        m_min = m_flat[0]
        m_max = m_flat[0]
        for i in range(1, n):
            v = r_flat[i]
            if v < r_min:
                r_min = v
            if v > r_max:
                r_max = v
            v = m_flat[i]
            if v < m_min:
                m_min = v
            if v > m_max:
                m_max = v

        r_range = r_max - r_min
        m_range = m_max - m_min
        if r_range < 1e-12:
            r_range = 1.0
        if m_range < 1e-12:
            m_range = 1.0

        r_scale = (bins - 1) / r_range
        m_scale = (bins - 1) / m_range

        for i in range(n):
            ri = int((r_flat[i] - r_min) * r_scale)
            mi = int((m_flat[i] - m_min) * m_scale)
            if ri < 0:
                ri = 0
            elif ri >= bins:
                ri = bins - 1
            if mi < 0:
                mi = 0
            elif mi >= bins:
                mi = bins - 1
            hist[ri, mi] += 1.0

        return hist

    @numba.njit(cache=True)
    def _mi_from_histogram(hist: np.ndarray) -> float:
        """Compute MI = H(ref) + H(mov) - H(ref, mov) from a joint histogram."""
        bins = hist.shape[0]
        total = 0.0
        for i in range(bins):
            for j in range(bins):
                total += hist[i, j]
        if total < 1e-12:
            return 0.0

        px = np.zeros(bins, dtype=np.float64)
        py = np.zeros(bins, dtype=np.float64)
        for i in range(bins):
            for j in range(bins):
                px[i] += hist[i, j]
                py[j] += hist[i, j]

        inv_total = 1.0 / total
        eps = 1e-300

        hx = 0.0
        for i in range(bins):
            p = px[i] * inv_total
            if p > eps:
                hx -= p * np.log(p)

        hy = 0.0
        for j in range(bins):
            p = py[j] * inv_total
            if p > eps:
                hy -= p * np.log(p)

        hxy = 0.0
        for i in range(bins):
            for j in range(bins):
                p = hist[i, j] * inv_total
                if p > eps:
                    hxy -= p * np.log(p)

        return hx + hy - hxy

    @numba.njit(cache=True)
    def _compute_mi_at_shift(
        ref: np.ndarray, mov: np.ndarray, dz: int, dy: int, dx: int, bins: int
    ) -> float:
        """Compute MI for a single candidate shift (overlap + histogram + MI)."""
        nz = ref.shape[0]
        ny = ref.shape[1]
        nx = ref.shape[2]

        # Overlap ranges along each axis.
        if dz >= 0:
            rz0, rz1, mz0, mz1 = dz, nz, 0, nz - dz
        else:
            rz0, rz1, mz0, mz1 = 0, nz + dz, -dz, nz

        if dy >= 0:
            ry0, ry1, my0, my1 = dy, ny, 0, ny - dy
        else:
            ry0, ry1, my0, my1 = 0, ny + dy, -dy, ny

        if dx >= 0:
            rx0, rx1, mx0, mx1 = dx, nx, 0, nx - dx
        else:
            rx0, rx1, mx0, mx1 = 0, nx + dx, -dx, nx

        if rz1 <= rz0 or ry1 <= ry0 or rx1 <= rx0:
            return -np.inf

        ref_sub = ref[rz0:rz1, ry0:ry1, rx0:rx1]
        mov_sub = mov[mz0:mz1, my0:my1, mx0:mx1]

        r_flat = ref_sub.ravel()
        m_flat = mov_sub.ravel()

        hist = _joint_histogram(r_flat, m_flat, bins)
        return _mi_from_histogram(hist)

    @numba.njit(parallel=True, cache=True)
    def _parallel_grid_search(
        ref: np.ndarray,
        mov: np.ndarray,
        shifts: np.ndarray,
        bins: int,
    ) -> np.ndarray:
        """Evaluate MI for every candidate shift in *shifts* using all CPU cores.

        Parameters
        ----------
        ref, mov : (Z, Y, X) float64 arrays
        shifts : (N, 3) int64 array of (dz, dy, dx) candidates
        bins : histogram bins

        Returns
        -------
        mi_values : (N,) float64 array of MI scores
        """
        n = shifts.shape[0]
        mi_values = np.empty(n, dtype=np.float64)
        for i in numba.prange(n):
            mi_values[i] = _compute_mi_at_shift(
                ref, mov, shifts[i, 0], shifts[i, 1], shifts[i, 2], bins
            )
        return mi_values

    _HAVE_NUMBA = True
    logger.debug("numba available – parallel MI grid search enabled")

except ImportError:
    _HAVE_NUMBA = False
    logger.debug("numba not installed – falling back to serial MI grid search")


# ---------------------------------------------------------------------------
# Pure-numpy fallback (unchanged logic from original implementation)
# ---------------------------------------------------------------------------

def _compute_mi(ref: np.ndarray, mov: np.ndarray, bins: int = _MI_BINS) -> float:
    """Compute mutual information between two arrays using a joint histogram."""
    r = ref.ravel().astype(np.float64)
    m = mov.ravel().astype(np.float64)

    hist_2d, _, _ = np.histogram2d(r, m, bins=bins)
    pxy = hist_2d / hist_2d.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

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
    """Return the overlapping sub-arrays after applying shift (dz, dy, dx) to *mov*."""
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


# ---------------------------------------------------------------------------
# Helpers for building shift arrays
# ---------------------------------------------------------------------------

def _build_shifts_array(sr_xy: int, sr_z: int, step: int) -> np.ndarray:
    """Pre-generate all (dz, dy, dx) candidates as an (N, 3) int64 array."""
    shifts = []
    for dz in range(-sr_z, sr_z + 1, step):
        for dy in range(-sr_xy, sr_xy + 1, step):
            for dx in range(-sr_xy, sr_xy + 1, step):
                shifts.append((dz, dy, dx))
    return np.array(shifts, dtype=np.int64)


def _build_fine_shifts_array(
    best_shift: tuple[int, int, int],
    fine_radius: int,
    sr_xy: int,
    sr_z: int,
) -> np.ndarray:
    """Pre-generate fine-pass shifts around *best_shift*."""
    bz, by, bx = best_shift
    shifts = []
    for dz in range(bz - fine_radius, bz + fine_radius + 1):
        if abs(dz) > sr_z:
            continue
        for dy in range(by - fine_radius, by + fine_radius + 1):
            if abs(dy) > sr_xy:
                continue
            for dx in range(bx - fine_radius, bx + fine_radius + 1):
                if abs(dx) > sr_xy:
                    continue
                shifts.append((dz, dy, dx))
    return np.array(shifts, dtype=np.int64) if shifts else np.empty((0, 3), dtype=np.int64)


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
        ref_f = np.ascontiguousarray(ref, dtype=np.float64)
        mov_f = np.ascontiguousarray(mov, dtype=np.float64)

        if _HAVE_NUMBA:
            return self._register_cpu_numba(ref_f, mov_f, sr_xy, sr_z)
        return self._register_cpu_serial(ref_f, mov_f, sr_xy, sr_z)

    def _register_cpu_numba(
        self,
        ref: np.ndarray,
        mov: np.ndarray,
        sr_xy: int,
        sr_z: int,
    ) -> RegistrationResult:
        """Numba-accelerated parallel coarse-to-fine MI search."""
        # ---- coarse pass (parallel) ------------------------------------
        coarse_shifts = _build_shifts_array(sr_xy, sr_z, _COARSE_STEP)
        coarse_mi = _parallel_grid_search(ref, mov, coarse_shifts, _MI_BINS)

        best_idx = int(np.argmax(coarse_mi))
        best_shift = (
            int(coarse_shifts[best_idx, 0]),
            int(coarse_shifts[best_idx, 1]),
            int(coarse_shifts[best_idx, 2]),
        )

        # ---- fine pass (parallel) --------------------------------------
        fine_radius = min(_FINE_RADIUS, sr_xy, sr_z)
        fine_shifts = _build_fine_shifts_array(best_shift, fine_radius, sr_xy, sr_z)

        if fine_shifts.shape[0] > 0:
            fine_mi = _parallel_grid_search(ref, mov, fine_shifts, _MI_BINS)
            fine_best_idx = int(np.argmax(fine_mi))
            best_shift = (
                int(fine_shifts[fine_best_idx, 0]),
                int(fine_shifts[fine_best_idx, 1]),
                int(fine_shifts[fine_best_idx, 2]),
            )
            peak_mi = float(fine_mi[fine_best_idx])
            all_mi = np.concatenate([coarse_mi, fine_mi])
        else:
            peak_mi = float(coarse_mi[best_idx])
            all_mi = coarse_mi

        confidence = self._compute_confidence(peak_mi, list(all_mi))

        return RegistrationResult(
            shift_x=best_shift[2],
            shift_y=best_shift[1],
            shift_z=best_shift[0],
            confidence=confidence,
            raw_metric_value=peak_mi,
            algorithm_name=ALGORITHM_NAME,
        )

    def _register_cpu_serial(
        self,
        ref: np.ndarray,
        mov: np.ndarray,
        sr_xy: int,
        sr_z: int,
    ) -> RegistrationResult:
        """Pure-numpy serial fallback (no numba)."""
        # ---- coarse pass ------------------------------------------------
        coarse_step = _COARSE_STEP
        best_shift, mi_values = self._grid_search(
            ref, mov, sr_xy, sr_z, coarse_step
        )

        # ---- fine pass --------------------------------------------------
        fine_radius = min(_FINE_RADIUS, sr_xy, sr_z)
        fine_shifts: list[tuple[int, int, int, float]] = []
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
                    overlap = _overlapping_regions(ref, mov, dz, dy, dx)
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
