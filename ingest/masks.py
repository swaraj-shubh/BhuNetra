"""Quality masks (DESIGN.md §3, §9). Classes and thresholds come from configs/."""

import numpy as np

from configs.settings import T, bands


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    """Binary dilation with a square kernel, numpy only."""
    if px <= 0:
        return mask
    out = mask.copy()
    padded = np.pad(mask, px)
    h, w = mask.shape
    for dy in range(-px, px + 1):
        for dx in range(-px, px + 1):
            out |= padded[px + dy : px + dy + h, px + dx : px + dx + w]
    return out


def s2_valid_mask(scl: np.ndarray, cld: np.ndarray | None = None) -> np.ndarray:
    """True where usable: SCL not invalid, Sen2Cor cloud probability below CLOUD_PROB; clouds dilated."""
    cloudy = np.isin(scl, bands["scl"]["cloud"])
    if cld is not None:
        cloudy |= cld >= T.quality.CLOUD_PROB * 100  # CLD is 0-100
    invalid = np.isin(scl, bands["scl"]["invalid"])
    return ~(invalid | dilate(cloudy, T.quality.CLOUD_DILATE_PX))
