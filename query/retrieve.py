"""Query path (DESIGN.md §3): parsed query -> branches -> fusion -> unique-site dedup -> ranked sites.

Branches: semantic (Qdrant text_vec) | change events (stored rows, SQL) | spatial (precomputed tile_context).
Only the query text is encoded here (§2.1). Events are read as stored: confidence is never recomputed (§2.2).
Module boundary (§5): reads events through the database, never through change/ internals.
"""

import json
from dataclasses import dataclass
from typing import Any

from qdrant_client import models as qm
from sqlalchemy import text

from configs.settings import T, fusion
from index import db, qdrant
from query.dedup import Site, dedup
from query.fuse import Candidate, fuse
from query.parser import Parsed

INDEXED_SENSORS = ("sentinel2",)  # text_vec exists for S2 composites only (Landsat/S1: M9)


@dataclass(frozen=True)
class Request:
    aoi: str | None
    date_range: tuple[str | None, str | None] | None
    k: int
    sensor: str | None = None  # change branch: only events this sensor supports (change_events.sensors)


def _pl(h: qm.ScoredPoint) -> dict[str, Any]:
    return h.payload or {}


def _date_filter(date_range) -> list[Any]:
    if not date_range:
        return []
    lo, hi = date_range
    out: list[Any] = []
    if lo:
        out.append(qm.FieldCondition(key="end_date", range=qm.DatetimeRange(gte=lo)))
    if hi:
        out.append(qm.FieldCondition(key="start_date", range=qm.DatetimeRange(lte=hi)))
    return out


def _best_per_tile(hits) -> dict[str, qm.ScoredPoint]:
    best: dict[str, qm.ScoredPoint] = {}
    for h in sorted(hits, key=lambda h: (-h.score, _pl(h)["tile_id"], _pl(h)["season"])):
        best.setdefault(_pl(h)["tile_id"], h)
    return best


def semantic(vec, aoi: str | None, date_range, using: str = "text_vec") -> dict[str, qm.ScoredPoint]:
    must: list[Any] = [qm.FieldCondition(key="aoi", match=qm.MatchValue(value=aoi))] if aoi else []
    must += _date_filter(date_range)
    return _best_per_tile(qdrant.search(vec, using, T.retrieval.TOP_N, qm.Filter(must=must)))


def semantic_for_tiles(vec, tiles: list[str]) -> dict[str, qm.ScoredPoint]:
    """Similarity of given tiles (e.g. event tiles outside the semantic top-N), best season each."""
    if not tiles:
        return {}
    flt = qm.Filter(must=[qm.FieldCondition(key="tile_id", match=qm.MatchAny(any=tiles))])
    return _best_per_tile(qdrant.search_best_per_tile(vec, "text_vec", len(tiles), flt))


def events(conn, p: Parsed, aoi: str | None, date_range, sensor: str | None = None) -> dict[str, list[dict]]:
    """Stored events matching the change intent, above the operating threshold (precision first, §2.11)."""
    where = ["e.confidence >= :c"]
    params: dict[str, Any] = {"c": T.events.OPERATING_CONF, "n": T.retrieval.TOP_N}
    if p.change_types:
        where.append("e.change_type::text = ANY(:types)")
        params["types"] = list(p.change_types)
    elif T.events.HIDDEN_TYPES:  # "any change": not the types that never reach TARGET_PRECISION
        where.append("NOT (e.change_type::text = ANY(:hidden))")
        params["hidden"] = list(T.events.HIDDEN_TYPES)
    if aoi:
        where.append("t.aoi = :aoi")
        params["aoi"] = aoi
    if date_range and date_range[0]:
        where.append("e.earliest_supported_date >= CAST(:t1 AS date)")
        params["t1"] = date_range[0]
    if date_range and date_range[1]:
        where.append("e.earliest_supported_date <= CAST(:t2 AS date)")
        params["t2"] = date_range[1]
    if sensor:
        where.append(":sensor = ANY(e.sensors)")
        params["sensor"] = sensor
    rows = conn.execute(
        text(
            "SELECT e.event_id, e.tile_id, e.change_type::text AS change_type, e.confidence, "
            "e.earliest_supported_date, e.source_scene_ids, e.cascade_version FROM change_events e "
            f"JOIN tiles t USING (tile_id) WHERE {' AND '.join(where)} "
            "ORDER BY e.confidence DESC, e.earliest_supported_date DESC, e.event_id LIMIT :n"
        ),
        params,
    ).mappings()
    by_tile: dict[str, list[dict]] = {}
    for r in rows:
        by_tile.setdefault(r["tile_id"], []).append(dict(r))
    return by_tile


def search(p: Parsed, req: Request, vec=None, exclude_tiles: frozenset[str] = frozenset()) -> dict:
    """vec: override the query embedding (Rocchio feedback); exclude_tiles: sites the analyst rejected."""
    from features import embed_clip  # torch import only when searching

    aoi = req.aoi or p.aoi
    date_range = req.date_range or p.date_range
    warnings = list(p.warnings)
    sensor = req.sensor or p.sensor
    if (
        sensor and sensor not in INDEXED_SENSORS and not p.change
    ):  # change events carry every supporting sensor
        warnings.append(f"sensor_not_indexed:{sensor}")
    if vec is None:
        vec = embed_clip.embed_text(p.semantic)

    with db.engine().connect() as conn:
        near = _available_near(conn, p.near, aoi, warnings)
        ev = events(conn, p, aoi, date_range, req.sensor or p.sensor) if p.change else {}
        if p.change:  # change intent: candidates are sites with a matching stored event, re-ranked by meaning
            sem = semantic_for_tiles(vec, sorted(ev))
            tiles = sorted(ev)
        else:
            sem = semantic(vec, aoi, date_range)
            tiles = sorted(sem)
        tiles = [t for t in tiles if t not in exclude_tiles]
        if not tiles:
            return {"sites": [], "warnings": warnings}
        sites = _assemble(conn, tiles, sem, ev, near, req.k, p.landcover)
    return {"sites": sites, "warnings": warnings}


def _available_near(conn, near: dict[str, float], aoi: str | None, warnings: list[str]) -> dict[str, float]:
    """Drop spatial targets with no precomputed context in scope (e.g. an organiser AOI without JRC / OSM staged: all distances are Infinity):
    filtering on them would silently return nothing. The caller reports each dropped target as a warning."""
    if not near:
        return near
    have = {
        r[0]
        for r in conn.execute(
            text(
                "SELECT DISTINCT c.target FROM tile_context c JOIN tiles t USING (tile_id) "
                "WHERE c.dist_m < 'Infinity' AND (CAST(:a AS text) IS NULL OR t.aoi = :a)"
            ),
            {"a": aoi},
        )
    }
    warnings += [f"spatial_context_unavailable:{t}" for t in near if t not in have]
    return {t: d for t, d in near.items() if t in have}


def landcover_share(fracs_by_year: dict[int, dict], year: int | None, groups: tuple[str, ...]) -> float:
    """Share of the named land-cover groups in the map year closest to the result's season (0 without maps)."""
    if not fracs_by_year or not groups:
        return 0.0
    y = min(fracs_by_year, key=lambda k: abs(k - year)) if year else max(fracs_by_year)
    return round(sum(fracs_by_year[y].get(g, 0.0) for g in groups), 4)


def _assemble(
    conn, tiles: list[str], sem: dict, ev: dict, near: dict[str, float], k: int, cover: tuple[str, ...] = ()
) -> list[dict]:
    """Candidates -> fusion -> unique-site dedup -> response dicts (shared by text and image search)."""
    rows = conn.execute(
        text(
            "SELECT t.tile_id, t.aoi, t.bounds_m, ST_AsGeoJSON(t.geom) AS geom, "
            "COALESCE(jsonb_object_agg(c.target, c.dist_m) FILTER (WHERE c.target IS NOT NULL), '{}') AS dist "
            "FROM tiles t LEFT JOIN tile_context c USING (tile_id) WHERE t.tile_id = ANY(:ids) "
            "GROUP BY t.tile_id, t.aoi, t.bounds_m, t.geom"
        ),
        {"ids": tiles},
    ).mappings()
    tile_rows = {r["tile_id"]: r for r in rows}
    comp_ids = [_pl(h)["comp_id"] for h in sem.values()]
    prov = {
        r["comp_id"]: r
        for r in conn.execute(
            text(
                "SELECT comp_id::text AS comp_id, model_versions, lineage->'observations' AS obs "
                "FROM composites WHERE comp_id = ANY(CAST(:ids AS uuid[]))"
            ),
            {"ids": comp_ids},
        ).mappings()
    }

    lc_rows: dict[str, dict[int, dict]] = {}
    if cover:  # area-type query: independent land-cover shares, a stored lookup (§2.1)
        for tid, year, fracs in conn.execute(
            text("SELECT tile_id, year, fracs FROM tile_landcover WHERE tile_id = ANY(:ids)"), {"ids": tiles}
        ):
            lc_rows.setdefault(tid, {})[year] = fracs
    cands = []
    for tid in tiles:
        tr = tile_rows[tid]
        b = tr["bounds_m"]
        h = sem.get(tid)
        tile_events = ev.get(tid, [])
        cands.append(
            Candidate(
                tile_id=tid,
                aoi=tr["aoi"],
                centroid_m=((b[0] + b[2]) / 2, (b[1] + b[3]) / 2),
                sim=h.score if h else None,
                event_conf=max((e["confidence"] for e in tile_events), default=0.0),
                q=float(_pl(h)["q_mean"]) if h else 0.0,
                dist_m={tgt: float(v) for tgt, v in (tr["dist"] or {}).items()},
                lc=landcover_share(
                    lc_rows.get(tid, {}),
                    int(_pl(h)["season"][:4]) if h and _pl(h).get("season") else None,
                    cover,
                ),
                payload={"hit": h, "events": tile_events, "geom": tr["geom"]},
            )
        )
    sites = dedup(fuse(cands, near, fusion), T.retrieval.DEDUP_RADIUS_M)[:k]
    changed = [s.head.cand.tile_id for s in sites if any(m.cand.payload["events"] for m in s.members)]
    latest = dict(  # change sites show their current state (latest dry season), not the best-matching season
        conn.execute(
            text(
                "SELECT DISTINCT ON (tile_id) tile_id, season FROM composites WHERE tile_id = ANY(:t) "
                "ORDER BY tile_id, season LIKE '%JJAS', end_date DESC"
            ),
            {"t": changed},
        ).all()
    )
    return [_site(rank, s, prov, latest.get(s.head.cand.tile_id)) for rank, s in enumerate(sites, 1)]


def _site(rank: int, site: Site, prov: dict, shown_season: str | None = None) -> dict:
    head = site.head
    h = head.cand.payload["hit"]
    evs = [e for m in site.members for e in m.cand.payload["events"]]
    comp = prov.get(_pl(h)["comp_id"]) if h else None
    season = shown_season or (_pl(h)["season"] if h else None)
    scene_ids = sorted(
        {o["scene_id"] for o in (comp["obs"] if comp else [])}
        | {s for e in evs for s in e["source_scene_ids"]}
    )
    return {
        "rank": rank,
        "tile_id": head.cand.tile_id,
        "aoi": head.cand.aoi,
        "score": head.score,
        "parts": head.parts,
        "season": season,
        "matched_season": _pl(h)["season"] if h else None,
        "q_mean": head.cand.q if h else None,
        "chip_url": f"/chips/{head.cand.tile_id}/{season}.png" if season else None,
        "geometry": json.loads(head.cand.payload["geom"]),
        "member_tiles": [m.cand.tile_id for m in site.members],
        "dist_m": head.cand.dist_m,
        "events": [
            {
                "event_id": e["event_id"],
                "tile_id": e.get("tile_id"),
                "change_type": e["change_type"],
                "confidence": e["confidence"],
                "earliest_supported_date": str(e["earliest_supported_date"]),
            }
            for e in evs
        ],
        "provenance": {
            "scene_ids": scene_ids,
            "model_versions": comp["model_versions"] if comp else {},
            "cascade_version": sorted({e["cascade_version"] for e in evs}) or None,
        },
    }


def search_image(
    vec, aoi: str | None, date_range, k: int, exclude_tiles: frozenset[str] = frozenset()
) -> list[dict]:
    """eo_vec similarity (DESIGN.md §2.3: never mixed with text_vec) -> same fusion/dedup as text search."""
    sem = semantic(vec, aoi, date_range, using="eo_vec")
    tiles = [t for t in sorted(sem) if t not in exclude_tiles]
    if not tiles:
        return []
    with db.engine().connect() as conn:
        return _assemble(conn, tiles, sem, {}, {}, k)


def overlapping(tile_id: str) -> frozenset[str]:
    """The tile and every grid tile sharing ground with it (TILE_M = 2 x STRIDE_M: row/col within 1). Image search
    excludes them: a half-overlapping neighbour "looks like" the seed trivially and is no discovery (PS 2.2.4)."""
    head, row, col = tile_id.rsplit("_", 2)
    r, c = int(row), int(col)
    return frozenset(f"{head}_{r + i:04d}_{c + j:04d}" for i in (-1, 0, 1) for j in (-1, 0, 1))


def eo_vector(conn, tile_id: str, season: str | None) -> tuple[list[float], str] | None:
    """Stored eo_vec of a tile (given season, else its latest dry-season one: monsoon composites are patchy)."""
    row = conn.execute(
        text(
            "SELECT eo_vec, season FROM composites WHERE tile_id = :t AND eo_vec IS NOT NULL "
            "AND (CAST(:s AS text) IS NULL OR season = :s) "
            "ORDER BY season LIKE '%JJAS', end_date DESC LIMIT 1"
        ),
        {"t": tile_id, "s": season},
    ).first()
    return (list(row[0]), row[1]) if row else None
