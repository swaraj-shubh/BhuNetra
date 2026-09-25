"""Radiometric standardisation (DESIGN.md §9)."""

import numpy as np


def to_reflectance(dn: np.ndarray, scale: float, offset: float, nodata: float | None = 0) -> np.ndarray:
    """DN -> reflectance with the asset's raster:bands scale/offset (from scene.json).

    S2 L2A baseline >= 04.00 carries offset -0.1 (the -1000 DN BOA_ADD_OFFSET). Never hardcode it by date.
    """
    refl = dn.astype(np.float32) * scale + offset
    if nodata is not None:
        refl[dn == nodata] = np.nan
    return refl


def db_to_linear(db_x100: np.ndarray, nodata: int = -32768) -> np.ndarray:
    """Staged S1 int16 dB*100 -> linear power (NaN where nodata). Speckle filtering works in linear."""
    lin = np.power(10.0, db_x100.astype(np.float32) / 1000.0)
    lin[db_x100 == nodata] = np.nan
    return lin


def rgb_quicklook(r: np.ndarray, g: np.ndarray, b: np.ndarray, vmax: float = 0.3) -> np.ndarray:
    """uint8 HxWx3 true colour, linear stretch 0..vmax with gamma 1.4; NaN -> black. Display only."""
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(np.nan_to_num(rgb, nan=0.0) / vmax, 0, 1) ** (1 / 1.4)
    return (rgb * 255).astype(np.uint8)
