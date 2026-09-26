"""Break pixels -> change events (DESIGN.md §2.2, §7). Pure over arrays + a reader for the timeline evidence.

Events group connected break pixels whose break falls in the same season (so a mine that grows every year becomes
one event per season, each with its own earliest date). Components smaller than MIN_EVENT_PX are dropped.

Confidence here is an **uncalibrated rule score** (coherence x size x magnitude). M5 replaces it with isotonic
calibration through a versioned re-run (§2.2); `evidence.calibrated` says which one an event carries.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import rasterio.features
from affine import Affine
from numpy.typing import NDArray
from pyproj import Transformer
from shapely.geometry import MultiPolygon, shape
from shapely.ops import transform as shp_transform

from change.classify import Evidence, classify
from configs.settings import T as CFG
from features.composites import season_of

CASCADE_VERSION = "m3-rules-0.1"  # bump on any change to harmonic/breaks/classify/events logic or thresholds

F32 = NDArray[np.float32]


@dataclass
class Event:
    pixels: NDArray[np.bool_]  # (H, W) footprint
    geom_wkt: str  # MultiPolygon, EPSG:4326
    centroid_m: tuple[float, float]
    change_type: str
    direction: str
    confidence: float
    break_idx: int
    last_normal_idx: int
    evidence: dict[str, Any]


def _shape_stats(mask: NDArray[np.bool_]) -> tuple[float, float]:
    """(elongation = major/minor axis, mean width in px) from the pixel coordinates' principal axes."""
    rr, cc = np.nonzero(mask)
    if len(rr) < 3:
        return 1.0, float(len(rr))
    ev = np.sort(np.linalg.eigvalsh(np.cov(np.stack([rr, cc]).astype(float))))
    minor, major = np.sqrt(np.maximum(ev, 1e-9))
    length_px = 4 * major  # ~ full extent of a uniform strip along its major axis
    return float(major / minor), float(len(rr) / max(length_px, 1.0))


def _magnitude(e: Evidence) -> float:
    c = CFG.classify
    return min(
        1.0,
        max(
            abs(e.d_ndvi) / abs(c.CLR_DNDVI),
            e.d_ndbi / c.CONS_DNDBI,
            e.d_bsi / c.CLR_DBSI,
            abs(e.d_mndwi) / c.WATER_DMNDWI,
            0.0,
        ),
    )


def extract(pix: dict[str, F32], dates: list[str], transform: Affine, epsg: int) -> list[Event]:
    """pix: cascade BANDS as (H, W) arrays. Returns events (no timeline yet: see timeline())."""
    brk = pix["break_idx"]
    has = np.isfinite(brk)
    season = np.full(brk.shape, "", dtype=object)
    season[has] = [season_of(dates[int(i)]) for i in brk[has]]
    to_ll = Transformer.from_crs(epsg, 4326, always_xy=True).transform
    min_px = CFG.events.MIN_EVENT_PX
    events: list[Event] = []
    for s in sorted(set(season[has])):
        m = (season == s).astype(np.uint8)
        m = rasterio.features.sieve(m, size=min_px, connectivity=8)
        polys = [
            shape(g)
            for g, v in rasterio.features.shapes(m, mask=m.astype(bool), transform=transform, connectivity=8)
            if v == 1
        ]
        if not polys:
            continue
        labels = rasterio.features.rasterize(
            ((p, i + 1) for i, p in enumerate(polys)), out_shape=m.shape, transform=transform, dtype="int32"
        )
        for i, poly in enumerate(polys, 1):
            fp = (labels == i) & (m == 1)
            n = int(fp.sum())
            if n < min_px:
                continue
            bi = brk[fp]
            consensus = int(np.median(bi))
            coherence = float(np.mean(np.abs(bi - consensus) <= 1))
            elong, width = _shape_stats(fp)
            ev = Evidence(
                d_ndvi=float(np.nanmedian(pix["d_ndvi"][fp])),
                d_mndwi=float(np.nanmedian(pix["d_mndwi"][fp])),
                d_ndbi=float(np.nanmedian(pix["d_ndbi"][fp])),
                d_bsi=float(np.nanmedian(pix["d_bsi"][fp])),
                water_pre=float(np.nanmean(pix["water_pre"][fp])),
                water_post=float(np.nanmean(pix["water_post"][fp])),
                elongation=elong,
                width_px=width,
            )
            ctype, direction = classify(ev)
            size = min(1.0, float(np.sqrt(n / (4 * min_px))))
            conf = round(coherence * size * _magnitude(ev), 4)
            ln = pix["last_normal_idx"][fp]
            last_normal = int(np.nanmedian(ln)) if np.isfinite(ln).any() else -1
            geom = shp_transform(to_ll, poly)
            mp = geom if isinstance(geom, MultiPolygon) else MultiPolygon([geom])
            c = poly.centroid
            events.append(
                Event(
                    pixels=fp,
                    geom_wkt=mp.wkt,
                    centroid_m=(c.x, c.y),
                    change_type=ctype,
                    direction=direction,
                    confidence=conf,
                    break_idx=consensus,
                    last_normal_idx=last_normal,
                    evidence={
                        "deltas": {
                            "ndvi": ev.d_ndvi,
                            "mndwi": ev.d_mndwi,
                            "ndbi": ev.d_ndbi,
                            "bsi": ev.d_bsi,
                        },
                        "water_frac": {"pre": ev.water_pre, "post": ev.water_post},
                        "shape": {
                            "n_px": n,
                            "area_m2": n * abs(transform.a * transform.e),
                            "elongation": elong,
                            "width_px": width,
                        },
                        "coherence": coherence,
                        "season": s,
                        "confidence_components": {
                            "coherence": coherence,
                            "size": size,
                            "magnitude": _magnitude(ev),
                        },
                        "calibrated": False,
                    },
                )
            )
    return events


def timelines(
    footprints: list[NDArray[np.intp]], per_date: Iterable[dict[str, F32]]
) -> list[dict[str, list[float | None]]]:
    """Median of each index over each event footprint, per date (the analyst's timeline evidence).

    footprints: flat pixel indices per event. per_date: one full-AOI index dict per date, read once for all events.
    """
    out: list[dict[str, list[float | None]]] = [{} for _ in footprints]
    for ix in per_date:
        flat = {name: arr.ravel() for name, arr in ix.items()}
        for tl, fp in zip(out, footprints, strict=True):
            for name, arr in flat.items():
                v = arr[fp]
                v = v[np.isfinite(v)]
                tl.setdefault(name, []).append(round(float(np.median(v)), 4) if v.size else None)
    return out
