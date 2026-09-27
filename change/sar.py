"""Sentinel-1 evidence for stored change events (M9; DESIGN.md §2.4: SAR never enters optical models).

Log-ratio = median gamma0 after - median gamma0 before, in dB, over the event footprint, using acquisitions from ONE
relative orbit and pass direction (staging keeps one per AOI). Medians are taken in linear power, then converted.
Agreement is a coarse physical check per change type; types without a clear SAR signature make no claim.
"""

import numpy as np
from numpy.typing import NDArray

from configs.settings import T


def stack_median_db(stack: list[NDArray[np.int16]], mask: NDArray[np.bool_], nodata: int) -> float | None:
    """Median over dates x footprint pixels of int16 dB*100 rasters (linear-power median), in dB."""
    vals = [s[mask & (s != nodata)] for s in stack]
    v = np.concatenate(vals) if vals else np.empty(0)
    if v.size == 0:
        return None
    lin = np.power(10.0, v.astype(np.float64) / 1000.0)
    return float(10.0 * np.log10(np.median(lin)))


def log_ratio_db(before: float | None, after: float | None) -> float | None:
    return None if before is None or after is None else after - before


def agrees(change_type: str, dvv: float | None, dvh: float | None) -> bool | None:
    """True/False = S1 supports/contradicts the optical type; None = no data or no SAR signature for this type."""
    if dvv is None or dvh is None:
        return None
    m = T.sar.MIN_DB
    if change_type == "construction":
        return dvv >= m
    if change_type == "clearance":
        return dvh <= -m
    if change_type == "water":
        return abs(dvv) >= T.sar.WATER_MIN_DB
    return None
