"""Confidence metric computation and colormap mapping."""

from __future__ import annotations

# Confidence thresholds (tuneable constants).
CONFIDENCE_HIGH = 0.7
CONFIDENCE_LOW = 0.4

GUIDANCE_TEXT = (
    "Confidence > 0.7: Registration likely reliable\n"
    "Confidence 0.4\u20130.7: Review shifts carefully, consider manual adjustment\n"
    "Confidence < 0.4: Registration may be unreliable, manual specification recommended"
)


def confidence_color_rgb(confidence: float) -> tuple[int, int, int]:
    """Map a confidence score (0–1) to an RGB colour via the Inferno colormap.

    Returns an (R, G, B) tuple with values in [0, 255].
    """
    try:
        from matplotlib import cm

        rgba = cm.inferno(max(0.0, min(1.0, confidence)))
        return (int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))
    except ImportError:
        # Fallback: simple grey-to-yellow gradient if matplotlib missing.
        v = max(0.0, min(1.0, confidence))
        return (int(255 * v), int(200 * v), int(50 * (1 - v)))
