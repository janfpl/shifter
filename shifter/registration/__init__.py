"""Auto-registration algorithms for chromatic shift detection."""

from shifter.registration.base import (
    RegistrationAlgorithm,
    RegistrationResult,
    estimate_registration_bytes,
)
from shifter.registration.phase_correlation import PhaseCorrelation
from shifter.registration.mutual_information import (
    MutualInformationRegistration,
)
from shifter.registration.cross_correlation import ZNCCRegistration
from shifter.registration.preprocessing import preprocess
from shifter.registration.confidence import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    GUIDANCE_TEXT,
    confidence_color_rgb,
)
from shifter.registration.gpu_utils import (
    gpu_available,
    gpu_fail_reason,
    gpu_name,
)

# Algorithm registry: display name → class.
ALGORITHM_REGISTRY: dict[str, type[RegistrationAlgorithm]] = {
    "Phase Cross-Correlation": PhaseCorrelation,
    "Mutual Information": MutualInformationRegistration,
    "Zero-Normalized Cross-Correlation": ZNCCRegistration,
}

# Maximum allowed search range (developer-tuneable constant).
MAX_SEARCH_RANGE = 200

__all__ = [
    "RegistrationAlgorithm",
    "RegistrationResult",
    "estimate_registration_bytes",
    "PhaseCorrelation",
    "MutualInformationRegistration",
    "ZNCCRegistration",
    "preprocess",
    "ALGORITHM_REGISTRY",
    "MAX_SEARCH_RANGE",
    "CONFIDENCE_HIGH",
    "CONFIDENCE_LOW",
    "GUIDANCE_TEXT",
    "confidence_color_rgb",
    "gpu_available",
    "gpu_fail_reason",
    "gpu_name",
]
