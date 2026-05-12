"""Beaufort-style wind speed colormap shared by renderer and UI legend."""

import numpy as np


# (max_speed_exclusive_m_s, rgba_float, label, range_text)
# Top band uses float('inf') as max.
BEAUFORT_BANDS = (
    (1.0,          (0.70, 0.85, 0.95, 0.9), "Calm",     "0-1 m/s"),
    (3.0,          (0.30, 0.60, 1.00, 0.9), "Light",    "1-3 m/s"),
    (6.0,          (0.30, 0.85, 0.50, 0.9), "Gentle",   "3-6 m/s"),
    (10.0,         (0.95, 0.85, 0.30, 0.9), "Moderate", "6-10 m/s"),
    (14.0,         (1.00, 0.60, 0.20, 0.9), "Fresh",    "10-14 m/s"),
    (20.0,         (1.00, 0.30, 0.30, 0.9), "Strong",   "14-20 m/s"),
    (float("inf"), (0.95, 0.30, 0.85, 0.9), "Gale+",    ">20 m/s"),
)

_BAND_THRESHOLDS = np.array(
    [band[0] for band in BEAUFORT_BANDS[:-1]], dtype=np.float32
)
_BAND_COLORS = np.array(
    [band[1] for band in BEAUFORT_BANDS], dtype=np.float32
)


def magnitudes_to_colors(magnitudes: np.ndarray) -> np.ndarray:
    """Map an (M,) array of speeds (m/s) to an (M, 4) RGBA float32 array."""
    if magnitudes.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    idx = np.searchsorted(_BAND_THRESHOLDS, magnitudes, side="right")
    return _BAND_COLORS[idx]
