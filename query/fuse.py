"""Fusion score (DESIGN.md §7): score = w_sim*sim_norm + w_conf*event_conf + w_q*q + w_sp*spatial_fit
(+ w_lc*landcover, amended 2026-09-28: independent land-cover share when the query names an area type).

Pure and deterministic: equal scores tie-break by tile_id, so a Qdrant rebuild ranks identically.
"""

from dataclasses import dataclass, field

from configs.settings import Fusion


@dataclass(frozen=True)
class Candidate:
    tile_id: str
    aoi: str
    centroid_m: tuple[float, float]  # in the AOI's UTM CRS (dedup distance)
    sim: float | None  # text_vec cosine; None = no semantic evidence for this tile
    event_conf: float  # best matching stored event confidence (fixed at ingest, §2.2); 0 = none
    q: float  # composite / observation quality
    dist_m: dict[str, float] = field(default_factory=dict)  # spatial target -> distance
    lc: float = 0.0  # land-cover share of the queried area type (features/landcover.py); 0 when none named
    payload: dict = field(
        default_factory=dict, compare=False
    )  # passthrough for the API (season, events, ...)


@dataclass(frozen=True)
class Scored:
    score: float
    cand: Candidate
    parts: dict[str, float]


def spatial_fit(dist_m: float, near_m: float) -> float:
    """1 within near_m, linear decay to 0 at 2*near_m."""
    if dist_m <= near_m:
        return 1.0
    return max(0.0, 1.0 - (dist_m - near_m) / near_m)


def fuse(cands: list[Candidate], near: dict[str, float], w: Fusion) -> list[Scored]:
    kept = []
    for c in cands:
        sp = min((spatial_fit(c.dist_m.get(t, float("inf")), d) for t, d in near.items()), default=0.0)
        if near and sp < 1.0:
            continue  # precision first (§2.11): "near X" is a hard limit; outside it the site is removed
        kept.append((c, sp))
    sims = [c.sim for c, _ in kept if c.sim is not None]  # normalise over survivors only
    lo, hi = (min(sims), max(sims)) if sims else (0.0, 0.0)
    out = []
    for c, sp in kept:
        sim_norm = 0.0 if c.sim is None else (1.0 if hi == lo else (c.sim - lo) / (hi - lo))
        parts = {
            "sim_norm": sim_norm,
            "event_conf": c.event_conf,
            "q": c.q,
            "spatial_fit": sp,
            "landcover": c.lc,
        }
        score = w.w_sim * sim_norm + w.w_conf * c.event_conf + w.w_q * c.q + w.w_sp * sp + w.w_lc * c.lc
        out.append(Scored(round(score, 6), c, {k: round(v, 4) for k, v in parts.items()}))
    return sorted(out, key=lambda s: (-s.score, s.cand.tile_id))
