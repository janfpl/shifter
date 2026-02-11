"""Abstract base class for registration algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


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

        Returns
        -------
        RegistrationResult
            Integer shifts and confidence score.
        """
