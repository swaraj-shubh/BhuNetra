"""Per-pixel change cascade over one AOI's index time series (DESIGN.md §3): harmonic fit -> persistent break ->
deseasonalised deltas. Runs at ingest time only (§2.1). Row blocks keep RAM bounded (§14.11).

Input: data/ard/<aoi>/<date>.tif (features/indices.py; unusable pixels are nodata).
Output: data/change/<aoi>/pixels.tif, float32 bands (BANDS), NaN = no value; break indices refer to `dates`.
"""

import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
from numpy.typing import NDArray
from rasterio.windows import Window

from change import breaks, harmonic
from features import indices

BANDS = ("break_idx", "last_normal_idx", "d_ndvi", "d_mndwi", "d_ndbi", "d_bsi", "water_pre", "water_post")
BLOCK_ROWS = 64  # ~135k px x ~80 dates x 4 indices per block: a few hundred MB

F32 = NDArray[np.float32]


@dataclass(frozen=True)
class Series:
    dates: list[str]  # ISO, ascending
    files: list[Path]
    t_days: NDArray[np.float64]


def series(ard_dir: Path) -> Series:
    files = sorted(ard_dir.glob("????-??-??.tif"))
    dates = [f.stem for f in files]
    d0 = date.fromisoformat(dates[0])
    t = np.array([(date.fromisoformat(d) - d0).days for d in dates], np.float64)
    return Series(dates=dates, files=files, t_days=t)


def block(stack: dict[str, NDArray[np.float64]], t: NDArray[np.float64]) -> dict[str, F32]:
    """stack: index -> (T, N) with NaN = unusable. Returns per-pixel BANDS (N,), pure (unit-testable)."""
    base = harmonic.baseline_mask(t)
    n = next(iter(stack.values())).shape[1]
    best = np.full(n, -1, np.int64)
    last_normal = np.full(n, -1, np.int64)
    resid: dict[str, NDArray[np.float64]] = {}
    for name, y in stack.items():
        valid = np.isfinite(y)
        m = harmonic.fit(y, valid & base[:, None], t)
        r = harmonic.residuals(y, valid, m, t)
        b = breaks.detect(r, ~base[:, None])
        earlier = (b.idx >= 0) & ((best < 0) | (b.idx < best))
        best[earlier] = b.idx[earlier]
        last_normal[earlier] = b.last_normal[earlier]
        resid[name] = np.where(valid, y - harmonic.predict(m, t), np.nan)  # deseasonalised anomaly

    post = (np.arange(len(t))[:, None] >= best[None, :]) & (best >= 0)[None, :]
    out: dict[str, F32] = {
        "break_idx": np.where(best >= 0, best, np.nan).astype(np.float32),
        "last_normal_idx": np.where(last_normal >= 0, last_normal, np.nan).astype(np.float32),
    }
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", RuntimeWarning
        )  # all-NaN slices -> NaN (pixel has no break / no data)
        for name in indices.NAMES:
            out[f"d_{name}"] = np.nanmedian(np.where(post, resid[name], np.nan), axis=0).astype(np.float32)
        mndwi = stack["mndwi"]
        pre_w = np.nanmedian(np.where(base[:, None], mndwi, np.nan), axis=0)
        post_w = np.nanmedian(np.where(post, mndwi, np.nan), axis=0)
    out["water_pre"] = np.where(np.isfinite(pre_w), pre_w > 0, np.nan).astype(np.float32)
    out["water_post"] = np.where(np.isfinite(post_w), post_w > 0, np.nan).astype(np.float32)
    return out


def run(ard_dir: Path, out_path: Path) -> Series:
    s = series(ard_dir)
    with rasterio.open(s.files[0]) as d0:
        prof, h, w = d0.profile, d0.height, d0.width
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".partial.tif")
    prof.update(
        driver="GTiff",
        dtype="float32",
        count=len(BANDS),
        nodata=np.nan,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    with rasterio.open(tmp, "w", **prof) as dst:
        for r0 in range(0, h, BLOCK_ROWS):
            win = Window(0, r0, w, min(BLOCK_ROWS, h - r0))
            reads = [indices.read(f, window=win) for f in s.files]
            stack = {
                n: np.stack([rd[n] for rd in reads]).reshape(len(s.files), -1).astype(np.float64)
                for n in indices.NAMES
            }
            res = block(stack, s.t_days)
            for i, b in enumerate(BANDS, 1):
                dst.write(res[b].reshape(int(win.height), w), i, window=win)
        for i, b in enumerate(BANDS, 1):
            dst.set_band_description(i, b)
    tmp.replace(out_path)
    return s
