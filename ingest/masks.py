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


def s2_cloud_mask(scl: np.ndarray, cld: np.ndarray | None = None) -> np.ndarray:
    """True on cloud/shadow (SCL classes or Sen2Cor CLD >= CLOUD_PROB), dilated by CLOUD_DILATE_PX."""
    cloudy = np.isin(scl, bands["scl"]["cloud"])
    if cld is not None:
        cloudy |= cld >= T.quality.CLOUD_PROB * 100  # CLD is 0-100
    return dilate(cloudy, T.quality.CLOUD_DILATE_PX)


def _bits(qa: np.ndarray, bits: list[int]) -> np.ndarray:
    return (qa.astype(np.uint16) & np.uint16(sum(1 << b for b in bits))) != 0


def landsat_cloud_mask(qa: np.ndarray) -> np.ndarray:
    """True on (dilated) cloud, cirrus or shadow from QA_PIXEL; USGS already dilates the cloud bit."""
    return _bits(qa, bands["landsat"]["qa_cloud_bits"])


def landsat_valid_mask(qa: np.ndarray) -> np.ndarray:
    """True where usable: not fill, cloud, cirrus, shadow or snow."""
    return ~_bits(qa, bands["landsat"]["qa_invalid_bits"])


def s2_valid_mask(scl: np.ndarray, cld: np.ndarray | None = None) -> np.ndarray:
    """True where usable: SCL not invalid and not (dilated) cloud."""
    return ~(np.isin(scl, bands["scl"]["invalid"]) | s2_cloud_mask(scl, cld))
