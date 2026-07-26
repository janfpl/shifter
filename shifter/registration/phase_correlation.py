"""Phase cross-correlation registration algorithm."""

from __future__ import annotations

import logging

import numpy as np

from shifter.registration.base import (
    ProgressCallback,
    RegistrationAlgorithm,
    RegistrationResult,
)

logger = logging.getLogger(__name__)

ALGORITHM_NAME = "Phase Cross-Correlation"


class PhaseCorrelation(RegistrationAlgorithm):
    """Integer-shift registration via phase cross-correlation.

    Uses ``skimage.registration.phase_cross_correlation`` on CPU, or a manual
    FFT-based implementation on GPU via cupy.

    Parameters
    ----------
    normalization : str or None
        Normalization mode passed to ``phase_cross_correlation``.
        ``"phase"`` (default) or ``None``.
    """

    def __init__(self, normalization: str | None = "phase") -> None:
        self.normalization = normalization

    def register(
        self,
        reference_volume: np.ndarray,
        moving_volume: np.ndarray,
        search_range_xy: int,
        search_range_z: int,
        use_gpu: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> RegistrationResult:
        # Phase correlation is a single FFT-based step with no internal search
        # loop to subdivide, so it reports completion only.
        if use_gpu:
            try:
                result = self._register_gpu(
                    reference_volume, moving_volume, search_range_xy, search_range_z
                )
                if progress_callback is not None:
                    progress_callback(1.0)
                return result
            except Exception as exc:
                logger.warning("GPU phase correlation failed (%s), falling back to CPU", exc)

        result = self._register_cpu(
            reference_volume, moving_volume, search_range_xy, search_range_z
        )
        if progress_callback is not None:
            progress_callback(1.0)
        return result

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
        from skimage.registration import phase_cross_correlation

        ref_f = ref.astype(np.float64)
        mov_f = mov.astype(np.float64)

        shift, error, phasediff = phase_cross_correlation(
            ref_f,
            mov_f,
            upsample_factor=1,
            normalization=self.normalization,
        )

        # shift is in (Z, Y, X) order and represents the correction to apply
        # to *moving* to align with *reference*.
        dz, dy, dx = int(round(shift[0])), int(round(shift[1])), int(round(shift[2]))

        # Enforce search range: if the detected shift exceeds the search
        # range, perform masked cross-correlation via FFT.
        if abs(dx) > sr_xy or abs(dy) > sr_xy or abs(dz) > sr_z:
            return self._register_cpu_masked(ref_f, mov_f, sr_xy, sr_z)

        raw = float(phasediff) if phasediff is not None else float(error)
        confidence = max(0.0, min(1.0, abs(raw)))

        return RegistrationResult(
            shift_x=dx,
            shift_y=dy,
            shift_z=dz,
            confidence=confidence,
            raw_metric_value=raw,
            algorithm_name=ALGORITHM_NAME,
        )

    def _register_cpu_masked(
        self,
        ref: np.ndarray,
        mov: np.ndarray,
        sr_xy: int,
        sr_z: int,
    ) -> RegistrationResult:
        """Masked FFT-based cross-correlation with search range enforcement.

        Spreads the FFT across the shared worker budget, which leaves a few
        cores free for the OS and the napari UI (see ``utils.worker_count``).
        """
        from scipy.fft import fftn, ifftn

        from shifter.utils import worker_count

        n_workers = worker_count()
        F_ref = fftn(ref, workers=n_workers)
        F_mov = fftn(mov, workers=n_workers)

        if self.normalization == "phase":
            cross_power = F_ref * np.conj(F_mov)
            eps = np.finfo(np.float64).eps
            cross_power /= np.maximum(np.abs(cross_power), eps)
        else:
            cross_power = F_ref * np.conj(F_mov)

        cc = np.real(ifftn(cross_power, workers=n_workers))

        # Build mask for allowed shifts.
        shape = ref.shape
        mask = np.zeros(shape, dtype=bool)
        for axis, sr in enumerate([sr_z, sr_xy, sr_xy]):
            idx = np.zeros(shape[axis], dtype=bool)
            idx[:sr + 1] = True
            idx[-(sr):] = True if sr > 0 else False
            slices = [slice(None)] * 3
            slices[axis] = idx
            if axis == 0:
                mask_axis = np.zeros(shape, dtype=bool)
                mask_axis[idx, :, :] = True
            elif axis == 1:
                mask_axis2 = np.zeros(shape, dtype=bool)
                mask_axis2[:, idx, :] = True
                mask_axis = mask_axis & mask_axis2 if axis > 0 else mask_axis2
            else:
                mask_axis3 = np.zeros(shape, dtype=bool)
                mask_axis3[:, :, idx] = True
                mask_axis = mask_axis & mask_axis3

        # Combined mask: intersection of all three axis masks.
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

        # Convert peak index to signed shift.
        shifts = []
        for p, s in zip(peak, shape):
            if p > s // 2:
                shifts.append(p - s)
            else:
                shifts.append(p)
        dz, dy, dx = shifts

        raw = float(cc[peak])
        confidence = max(0.0, min(1.0, abs(raw)))

        return RegistrationResult(
            shift_x=dx,
            shift_y=dy,
            shift_z=dz,
            confidence=confidence,
            raw_metric_value=raw,
            algorithm_name=ALGORITHM_NAME,
        )

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
        import cupy as cp
        from cupyx.scipy.fft import fftn, ifftn

        ref_g = cp.asarray(ref, dtype=cp.float64)
        mov_g = cp.asarray(mov, dtype=cp.float64)

        F_ref = fftn(ref_g)
        F_mov = fftn(mov_g)

        if self.normalization == "phase":
            cross_power = F_ref * cp.conj(F_mov)
            eps = cp.finfo(cp.float64).eps
            cross_power /= cp.maximum(cp.abs(cross_power), eps)
        else:
            cross_power = F_ref * cp.conj(F_mov)

        cc = cp.real(ifftn(cross_power))

        # Build search-range mask on GPU.
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
        confidence = max(0.0, min(1.0, abs(raw)))

        return RegistrationResult(
            shift_x=dx,
            shift_y=dy,
            shift_z=dz,
            confidence=confidence,
            raw_metric_value=raw,
            algorithm_name=ALGORITHM_NAME,
        )
