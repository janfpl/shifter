"""Abstract base class for registration algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np

# Optional per-registration progress hook. Called with a fraction in [0.0, 1.0]
# indicating how far *this* pairwise registration has progressed. Algorithms
# call it periodically (e.g. through a mutual-information grid search) so the UI
# can advance a progress bar within a single registration, not just between
# channels. May be omitted (None) by callers that don't need it.
ProgressCallback = Callable[[float], None]


@dataclass
class RegistrationResult:
    """Result of a pairwise registration.

    Attributes
    ----------
    shift_x : int
        Detected translation in X (voxels).
    shift_y : int
        Detected translation in Y (voxels).
    shift_z : int
        Detected translation in Z (voxels).
    confidence : float
        Confidence score mapped to [0.0, 1.0].
    raw_metric_value : float
        The raw value of the algorithm-specific quality metric before
        confidence mapping.
    algorithm_name : str
        Human-readable name of the algorithm that produced this result.
    """

    shift_x: int = 0
    shift_y: int = 0
    shift_z: int = 0
    confidence: float = 0.0
    raw_metric_value: float = 0.0
    algorithm_name: str = ""


# Conservative estimate of peak host-RAM bytes needed per voxel to run a
# given algorithm's CPU path. FFT-based algorithms (phase correlation,
# ZNCC) hold several float64/complex128 copies of the sub-volume
# simultaneously (skimage/scipy internals plus our own pre-cast copies);
# mutual information only needs raveled float64 copies plus small
# histograms. deedsBCV holds two 12-channel float32 descriptor volumes
# (96 B/voxel) plus the six-channel distance buffer and its box-filter
# temporaries while building each of them. These are upper-bound heuristics
# for pre-flight warnings, not an exact accounting.
MEMORY_BYTES_PER_VOXEL: dict[str, int] = {
    "Phase Cross-Correlation": 64,
    "Zero-Normalized Cross-Correlation": 64,
    "Mutual Information": 24,
    "deedsBCV (MIND-SSC)": 160,
}
_DEFAULT_MEMORY_BYTES_PER_VOXEL = 64


def estimate_registration_bytes(shape: tuple[int, int, int], algorithm_name: str) -> int:
    """Rough upper-bound estimate of peak CPU RAM (bytes) to register *shape*."""
    voxels = 1
    for d in shape:
        voxels *= d
    per_voxel = MEMORY_BYTES_PER_VOXEL.get(algorithm_name, _DEFAULT_MEMORY_BYTES_PER_VOXEL)
    return voxels * per_voxel


class RegistrationAlgorithm(ABC):
    """Interface that every registration algorithm must implement."""

    @abstractmethod
    def register(
        self,
        reference_volume: np.ndarray,
        moving_volume: np.ndarray,
        search_range_xy: int,
        search_range_z: int,
        use_gpu: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> RegistrationResult:
        """Detect the integer voxel shift that best aligns *moving_volume*
        to *reference_volume*.

        Parameters
        ----------
        reference_volume : np.ndarray
            3-D array (Z, Y, X) of the reference channel sub-volume.
        moving_volume : np.ndarray
            3-D array (Z, Y, X) of the channel to register.
        search_range_xy : int
            Maximum allowed shift magnitude in X and Y.
        search_range_z : int
            Maximum allowed shift magnitude in Z.
        use_gpu : bool
            If True and a GPU is available, run computations on GPU.
        progress_callback : ProgressCallback, optional
            If given, called periodically with a fraction in [0.0, 1.0] as this
            registration progresses. Algorithms with an internal search (e.g.
            mutual information) report fine-grained progress; single-shot
            algorithms report completion only.

        Returns
        -------
        RegistrationResult
            Integer shifts and confidence score.
        """
