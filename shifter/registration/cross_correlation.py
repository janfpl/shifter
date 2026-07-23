"""Zero-normalized cross-correlation (ZNCC) registration algorithm.

CPU FFT path uses multithreaded ``scipy.fft`` (``workers=-1``).
The brute-force fallback parallelises shift evaluations across CPU cores
using ``concurrent.futures.ThreadPoolExecutor`` (numpy releases the GIL
for the heavy number-crunching).
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from shifter.registration.base import (
    RegistrationAlgorithm,
    RegistrationResult,
)

logger = logging.getLogger(__name__)

ALGORITHM_NAME = "Zero-Normalized Cross-Correlation"


def _zncc_at_shift(
    ref: np.ndarray, mov: np.ndarray, dz: int, dy: int, dx: int
) -> float | None:
    """Compute ZNCC on the overlapping region for a given shift.

    Returns None if the overlap is too small.
    """
    nz, ny, nx = ref.shape

    def _range(shift: int, length: int):
        if shift >= 0:
            return (shift, length), (0, length - shift)
        return (0, length + shift), (-shift, length)

    (rz0, rz1), (mz0, mz1) = _range(dz, nz)
    (ry0, ry1), (my0, my1) = _range(dy, ny)
    (rx0, rx1), (mx0, mx1) = _range(dx, nx)

    if rz1 <= rz0 or ry1 <= ry0 or rx1 <= rx0:
        return None

    r = ref[rz0:rz1, ry0:ry1, rx0:rx1].astype(np.float64)
    m = mov[mz0:mz1, my0:my1, mx0:mx1].astype(np.float64)

    r_mean = r.mean()
    m_mean = m.mean()
    r_std = r.std()
    m_std = m.std()

    if r_std < 1e-12 or m_std < 1e-12:
        return 0.0

    n = r.size
    return float(np.sum((r - r_mean) * (m - m_mean)) / (n * r_std * m_std))


class ZNCCRegistration(RegistrationAlgorithm):
    """Registration via zero-normalized cross-correlation.

    Uses an FFT-based approach when the search range is moderate, or falls
    back to a brute-force evaluation for each candidate shift.
    """

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
                logger.warning("GPU ZNCC failed (%s), falling back to CPU", exc)

        return self._register_cpu(
            reference_volume, moving_volume, search_range_xy, search_range_z
        )

    # ------------------------------------------------------------------ #
    # CPU — FFT-based approach (multithreaded)
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

        shape = ref_f.shape

        # Try FFT-based approach first.
        try:
            return self._fft_zncc(ref_f, mov_f, shape, sr_xy, sr_z)
        except Exception:
            pass

        # Fallback to brute force.
        return self._brute_force(ref_f, mov_f, sr_xy, sr_z)

    def _fft_zncc(
        self,
        ref: np.ndarray,
        mov: np.ndarray,
        shape: tuple[int, ...],
        sr_xy: int,
        sr_z: int,
    ) -> RegistrationResult:
        """FFT-based ZNCC: cross-correlate in Fourier space, then normalise.

        Uses ``workers=-1`` to spread the FFT across all available CPU cores.
        """
        from scipy.fft import fftn, ifftn

        ref_zm = ref - ref.mean()
        mov_zm = mov - mov.mean()

        # Cross-correlation via FFT — use all CPU cores.
        F_ref = fftn(ref_zm, workers=-1)
        F_mov = fftn(mov_zm, workers=-1)
        cc = np.real(ifftn(F_ref * np.conj(F_mov), workers=-1))

        # Normalise by product of standard deviations and volume size.
        n = ref.size
        norm = n * ref.std() * mov.std()
        if norm < 1e-12:
            return RegistrationResult(algorithm_name=ALGORITHM_NAME)
        cc /= norm

        # Mask to search range.
        z_ok = np.zeros(shape[0], dtype=bool)
        z_ok[:sr_z + 1] = True
        if sr_z > 0:
            z_ok[-sr_z:] = True
        y_ok = np.zeros(shape[1], dtype=bool)
        y_ok[:sr_xy + 1] = True
        if sr_xy > 0:
            y_ok[-sr_xy:] = True
        x_ok = np.zeros(shape[2], dtype=bool)
        x_ok[:sr_xy + 1] = True
        if sr_xy > 0:
            x_ok[-sr_xy:] = True

        mask = z_ok[:, None, None] & y_ok[None, :, None] & x_ok[None, None, :]
        cc[~mask] = -np.inf

        peak = np.unravel_index(np.argmax(cc), shape)
        shifts = []
        for p, s in zip(peak, shape):
            shifts.append(p - s if p > s // 2 else p)
        dz, dy, dx = shifts

        raw = float(cc[peak])
        # Confidence: (ZNCC + 1) / 2 maps [-1, 1] → [0, 1].
        confidence = max(0.0, min(1.0, (raw + 1.0) / 2.0))

        return RegistrationResult(
            shift_x=dx,
            shift_y=dy,
            shift_z=dz,
            confidence=confidence,
            raw_metric_value=raw,
            algorithm_name=ALGORITHM_NAME,
        )

    def _brute_force(
        self,
        ref: np.ndarray,
        mov: np.ndarray,
        sr_xy: int,
        sr_z: int,
    ) -> RegistrationResult:
        """Evaluate ZNCC at every candidate shift, parallelised across CPU cores."""
        # Build list of all candidate shifts.
        candidates = [
            (dz, dy, dx)
            for dz in range(-sr_z, sr_z + 1)
            for dy in range(-sr_xy, sr_xy + 1)
            for dx in range(-sr_xy, sr_xy + 1)
        ]

        n_workers = min(os.cpu_count() or 1, len(candidates))

        def _eval_shift(args):
            dz, dy, dx = args
            val = _zncc_at_shift(ref, mov, dz, dy, dx)
            return (dz, dy, dx, val)

        best_zncc = -2.0
        best_shift = (0, 0, 0)

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for dz, dy, dx, val in pool.map(_eval_shift, candidates):
                if val is not None and val > best_zncc:
                    best_zncc = val
                    best_shift = (dz, dy, dx)

        confidence = max(0.0, min(1.0, (best_zncc + 1.0) / 2.0))

        return RegistrationResult(
            shift_x=best_shift[2],
            shift_y=best_shift[1],
            shift_z=best_shift[0],
            confidence=confidence,
            raw_metric_value=best_zncc,
            algorithm_name=ALGORITHM_NAME,
        )

    # ------------------------------------------------------------------ #
    # GPU path — FFT-based
    # ------------------------------------------------------------------ #

    def _register_gpu(
        self,
        ref: np.ndarray,
        mov: np.ndarray,
        sr_xy: int,
        sr_z: int,
    ) -> RegistrationResult:
        import cupy as cp
        from cupyx.scipy.fft import fftn, ifftn

        ref_g = cp.asarray(ref, dtype=cp.float64)
        mov_g = cp.asarray(mov, dtype=cp.float64)

        ref_zm = ref_g - ref_g.mean()
        mov_zm = mov_g - mov_g.mean()

        F_ref = fftn(ref_zm)
        F_mov = fftn(mov_zm)
        cc = cp.real(ifftn(F_ref * cp.conj(F_mov)))

        n = ref_g.size
        norm = float((n * ref_g.std() * mov_g.std()).get())
        if norm < 1e-12:
            return RegistrationResult(algorithm_name=ALGORITHM_NAME)
        cc /= norm

        shape = ref.shape
        z_ok = cp.zeros(shape[0], dtype=cp.bool_)
        z_ok[:sr_z + 1] = True
        if sr_z > 0:
            z_ok[-sr_z:] = True
        y_ok = cp.zeros(shape[1], dtype=cp.bool_)
        y_ok[:sr_xy + 1] = True
        if sr_xy > 0:
            y_ok[-sr_xy:] = True
        x_ok = cp.zeros(shape[2], dtype=cp.bool_)
        x_ok[:sr_xy + 1] = True
        if sr_xy > 0:
            x_ok[-sr_xy:] = True

        mask = z_ok[:, None, None] & y_ok[None, :, None] & x_ok[None, None, :]
        cc[~mask] = -cp.inf

        peak = cp.unravel_index(cp.argmax(cc), shape)
        peak = tuple(int(p.get()) for p in peak)

        shifts = []
        for p, s in zip(peak, shape):
            shifts.append(p - s if p > s // 2 else p)
        dz, dy, dx = shifts

        raw = float(cc[peak].get())
        confidence = max(0.0, min(1.0, (raw + 1.0) / 2.0))

        return RegistrationResult(
            shift_x=dx,
            shift_y=dy,
            shift_z=dz,
            confidence=confidence,
            raw_metric_value=raw,
            algorithm_name=ALGORITHM_NAME,
        )
