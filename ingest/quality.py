"""Per tile x observation quality score q (DESIGN.md §7). Observations with q < Q_MIN never enter a series."""

import numpy as np
from affine import Affine

from configs.settings import T
from ingest.tiling import Tile, window


def q_score(valid_frac, shift_px, cloud_near, sun_elev):
    """q = valid_frac * clip(1 - shift_px, 0, 1) * (1 - 0.5*cloud_near) * (1 - 0.3*[sun_elev < SUN_ELEV_MIN]).

    Vectorised. shift_px NaN (co-registration not run yet, M3) counts as 0.
    """
    shift = np.nan_to_num(np.asarray(shift_px, dtype=np.float64), nan=0.0)
    low_sun = np.asarray(sun_elev) < T.quality.SUN_ELEV_MIN
    return valid_frac * np.clip(1 - shift, 0, 1) * (1 - 0.5 * np.asarray(cloud_near)) * (1 - 0.3 * low_sun)


def _integral(mask: np.ndarray) -> np.ndarray:
    s = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=np.int64)
    s[1:, 1:] = mask.cumsum(0).cumsum(1)
    return s


def _box(s: np.ndarray, r0, c0, r1, c1) -> np.ndarray:
    """Sum of the mask over [r0:r1, c0:c1] for arrays of boxes, clipped to the raster."""
    h, w = s.shape[0] - 1, s.shape[1] - 1
    r0, r1 = np.clip(r0, 0, h), np.clip(r1, 0, h)
    c0, c1 = np.clip(c0, 0, w), np.clip(c1, 0, w)
    return s[r1, c1] - s[r0, c1] - s[r1, c0] + s[r0, c0]


def tile_fractions(
    valid: np.ndarray, cloud: np.ndarray, transform: Affine, tiles: list[Tile]
) -> tuple[np.ndarray, np.ndarray]:
    """(valid_frac, cloud_near) per tile, in the order of `tiles`.

    valid_frac: valid pixels / full tile area (parts outside the raster count as invalid).
    cloud_near: cloud/shadow fraction in the tile grown by STRIDE_M on every side (clipped to the raster).
    Definition of cloud_near is ours.
    """
    wins = [window(t, transform) for t in tiles]
    r0 = np.array([round(w.row_off) for w in wins])
    c0 = np.array([round(w.col_off) for w in wins])
    r1 = r0 + np.array([round(w.height) for w in wins])
    c1 = c0 + np.array([round(w.width) for w in wins])
    valid_frac = _box(_integral(valid), r0, c0, r1, c1) / ((r1 - r0) * (c1 - c0))

    b = round(T.tiling.STRIDE_M / abs(transform.a))
    h, w = cloud.shape
    nr0, nc0, nr1, nc1 = (
        np.clip(r0 - b, 0, h),
        np.clip(c0 - b, 0, w),
        np.clip(r1 + b, 0, h),
        np.clip(c1 + b, 0, w),
    )
    area = np.maximum((nr1 - nr0) * (nc1 - nc0), 1)
    cloud_near = _box(_integral(cloud), nr0, nc0, nr1, nc1) / area
    return valid_frac, cloud_near
