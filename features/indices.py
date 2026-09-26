"""Spectral indices per valid observation (DESIGN.md §3, §7) -> data/ard/<aoi>/<date>.tif (+ .json lineage).

    NDVI  = (nir - red) / (nir + red)
    MNDWI = (green - swir1) / (green + swir1)
    NDBI  = (swir1 - nir) / (swir1 + nir)
    BSI   = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue))

Stored as int16 x SCALE (nodata = NODATA). Unusable pixels (invalid, or in no tile with q >= Q_MIN) are nodata, so
low-quality observations never enter the change time series (§2.5).
"""

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine

NAMES = ("ndvi", "mndwi", "ndbi", "bsi")
SCALE = 10_000  # storage encoding: indices are in [-1, 1]
NODATA = -32768


def _nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # 0/0 -> NaN is the intended result
        out = (a - b) / (a + b)
    out[~np.isfinite(out) | (np.abs(out) > 1)] = np.nan  # |x| > 1 only arises from negative reflectance noise
    return out.astype(np.float32)


def compute(refl: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    r = refl
    return {
        "ndvi": _nd(r["nir"], r["red"]),
        "mndwi": _nd(r["green"], r["swir1"]),
        "ndbi": _nd(r["swir1"], r["nir"]),
        "bsi": _nd(r["swir1"] + r["red"], r["nir"] + r["blue"]),
    }


def write(
    path: Path,
    ix: dict[str, np.ndarray],
    usable: np.ndarray,
    transform: Affine,
    epsg: int,
    lineage: dict[str, Any],
) -> None:
    h, w = usable.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial.tif")
    prof = {
        "driver": "COG",
        "dtype": "int16",
        "count": len(NAMES),
        "height": h,
        "width": w,
        "crs": f"EPSG:{epsg}",
        "transform": transform,
        "nodata": NODATA,
        "compress": "deflate",
    }
    with rasterio.open(tmp, "w", **prof) as d:
        for i, n in enumerate(NAMES, 1):
            v = ix[n]
            enc = np.where(usable & np.isfinite(v), np.round(np.nan_to_num(v) * SCALE), NODATA).astype(
                np.int16
            )
            d.write(enc, i)
            d.set_band_description(i, n)
    tmp.replace(path)
    path.with_suffix(".json").write_text(json.dumps(lineage), encoding="utf-8")


def read(path: Path, window: Any = None) -> dict[str, np.ndarray]:
    """Decoded float32 indices, NaN = nodata."""
    with rasterio.open(path) as d:
        raw = d.read(window=window)
    out = np.where(raw == NODATA, np.nan, raw.astype(np.float32) / SCALE)
    return dict(zip(NAMES, out, strict=True))
