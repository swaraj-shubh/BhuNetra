"""Seasonal composites per AOI (DESIGN.md §3): per-pixel median of usable pixels, IMD seasons (configs/thresholds.yaml).

A pixel is usable when it is valid in its observation (masks) AND lies in at least one tile whose q >= Q_MIN (§2.5).
Output: data/composites/<aoi>/<season>.tif (7 canonical bands, float32 reflectance, COG) + <season>.json
(lineage + per-tile n_obs / q_mean / valid_frac, in tile order).
"""

import calendar
import json
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

from configs.settings import PROCESSING_VERSION, T, bands, paths
from ingest import quality, scene
from ingest.tiling import Tile, window

CANON = list(bands["sentinel2"]["bands"])


def season_of(iso_date: str) -> str:
    d = date.fromisoformat(iso_date)
    for name, months in T.composites.SEASONS.items():
        if d.month in months:
            # a season spanning New Year (DJF) belongs to the year of its last month
            return f"{d.year + (1 if d.month > months[-1] else 0)}-{name}"
    raise ValueError(f"month {d.month} not in any season")


def season_window(season: str) -> tuple[str, str]:
    year, name = int(season[:4]), season[5:]
    months = T.composites.SEASONS[name]
    first_year = year - 1 if months[0] > months[-1] else year
    last = calendar.monthrange(year, months[-1])[1]
    return f"{first_year}-{months[0]:02d}-01", f"{year}-{months[-1]:02d}-{last:02d}"


def q_ok_mask(q: np.ndarray, tiles: list[Tile], transform: Affine, shape: tuple[int, int]) -> np.ndarray:
    ok = np.zeros(shape, bool)
    for qi, t in zip(q, tiles, strict=True):
        if qi >= T.quality.Q_MIN:
            w = window(t, transform)
            r, c = round(w.row_off), round(w.col_off)
            ok[max(r, 0) : r + round(w.height), max(c, 0) : c + round(w.width)] = True
    return ok


def median_composite(
    refls: list[dict[str, np.ndarray]], usable: list[np.ndarray]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Per-band nanmedian over observations; returns (composite, n_obs per pixel)."""
    n = np.sum(usable, axis=0).astype(np.int16)
    out = {}
    for b in refls[0]:
        stack = np.stack([np.where(u, r[b], np.nan) for r, u in zip(refls, usable, strict=True)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # 'All-NaN slice': NaN is the intended result
            out[b] = np.nanmedian(stack, axis=0).astype(np.float32)
    return out, n


def build(aoi: str, season: str, folders: list[Path], tiles: list[Tile]) -> dict:
    """Composite one AOI x season from staged S2 folders. Returns the sidecar JSON dict (also written to disk)."""
    # holds every obs of the season in RAM (~120 MB each at 10 m, 1 per month staged -> <= 4); stream if
    # organiser data brings dense time series.
    refls, usable, per_tile_q, lineage = [], [], [], []
    transform = epsg = shape = None
    for f in folders:
        obs = scene.load(f)
        transform, epsg, shape = obs.transform, obs.epsg, obs.valid.shape
        vf, cn = quality.tile_fractions(obs.valid, obs.cloud, obs.transform, tiles)
        q = quality.q_score(vf, np.nan, cn, obs.sun_elev if obs.sun_elev is not None else np.nan)
        refls.append(obs.refl)
        usable.append(obs.valid & q_ok_mask(q, tiles, obs.transform, shape))
        per_tile_q.append(q)
        lineage.append({"scene_id": obs.scene_id, "fingerprint": scene.fingerprint(f), "date": obs.date})
    if not refls:
        raise ValueError(f"{aoi} {season}: no observations")

    comp, _ = median_composite(refls, usable)
    present = [b for b in CANON if b in comp]  # local imagery may lack bands (e.g. RGB-only)
    qs = np.stack(per_tile_q)  # (n_obs, n_tiles)
    good = qs >= T.quality.Q_MIN
    n_obs = good.sum(0)
    q_mean = np.where(n_obs > 0, (qs * good).sum(0) / np.maximum(n_obs, 1), 0.0)
    comp_valid = ~np.isnan(comp["red"])
    valid_frac, _ = quality.tile_fractions(comp_valid, np.zeros_like(comp_valid), transform, tiles)

    out_dir = paths.composites / aoi
    out_dir.mkdir(parents=True, exist_ok=True)
    tif = out_dir / f"{season}.tif"
    prof = {
        "driver": "COG",
        "dtype": "float32",
        "count": len(present),
        "height": shape[0],
        "width": shape[1],
        "crs": f"EPSG:{epsg}",
        "transform": transform,
        "nodata": np.nan,
        "compress": "deflate",
    }
    tmp = tif.with_suffix(".partial.tif")
    with rasterio.open(tmp, "w", **prof) as d:
        for i, b in enumerate(present, 1):
            d.write(comp[b], i)
            d.set_band_description(i, b)
    tmp.replace(tif)

    start, end = season_window(season)
    side = {
        "aoi": aoi,
        "season": season,
        "start_date": start,
        "end_date": end,
        "file": tif.name,
        "bands": present,
        "lineage": {
            "observations": lineage,
            "processing_version": PROCESSING_VERSION,
            "method": "per-pixel nanmedian of valid pixels inside tiles with q >= Q_MIN",
            "Q_MIN": T.quality.Q_MIN,
        },
        "tiles": {
            "tile_id": [t.tile_id for t in tiles],
            "n_obs": n_obs.tolist(),
            "q_mean": np.round(q_mean, 4).tolist(),
            "valid_frac": np.round(valid_frac, 4).tolist(),
        },
    }
    (out_dir / f"{season}.json").write_text(json.dumps(side), encoding="utf-8")
    return side
