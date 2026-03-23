"""GPU detection and cupy/numpy fallback logic.

Re-exported from registration.gpu_utils for use by the processing pipeline.
This module is the canonical import location for GPU utilities.
"""

from chromatic_shift_corrector.registration.gpu_utils import (  # noqa: F401
    gpu_available,
    gpu_fail_reason,
    gpu_name,
    get_compute_backend,
    to_device,
    to_numpy,
)
