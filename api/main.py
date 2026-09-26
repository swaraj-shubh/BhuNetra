"""FastAPI app (DESIGN.md §12). /stats, /search/text (M4: parser + fusion), /change, /events, map chips.

uvicorn api.main:app --port 8000
"""

import io
import json
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path

import rasterio
from fastapi import FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field
from pyproj import Transformer
from sqlalchemy import text

from configs.settings import T, aois, paths, sources
from index import db
from ingest import tiling
from ingest.standardise import rgb_quicklook

LATENCY_MS: dict[str, float] = {}  # last measured, per route (reported by /stats)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load CLIP and hash its weights once at startup: a cold first query took 26 s (offline run, 2026-09-26)."""
    from features import embed_clip

    try:
        embed_clip.embed_text("warm-up")
        embed_clip.version()
    except FileNotFoundError:  # no weights (CI, laptop without models/): /stats still works, search will 500
        pass
    yield


app = FastAPI(title="SIH26227 EO archive", lifespan=lifespan)

from api.label import router as label_router

app.include_router(label_router)


def _dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def _area_km2(bbox: tuple[float, float, float, float]) -> float:
    epsg = tiling.aoi_epsg(bbox)
    x0, y0, x1, y1 = Transformer.from_crs(4326, epsg, always_xy=True).transform_bounds(*bbox)
    return (x1 - x0) * (y1 - y0) / 1e6


@app.get("/stats")
def stats() -> dict:
    """Archive size and footprint. Latencies are not measured yet (M2+)."""
    per_aoi = {}
    for name, a in aois.items():
        epsg = tiling.aoi_epsg(a.bbox)
        scenes = {s: len(list((paths.raw / s / name).glob("*/scene.json"))) for s in sources["sources"]}
        per_aoi[name] = {
            "area_km2": round(_area_km2(a.bbox), 1),
            "tiles": sum(1 for _ in tiling.tiles(name, tiling.aoi_bounds_m(a.bbox, epsg), epsg)),
            "scenes": {s: n for s, n in scenes.items() if n},
        }
    report = paths.logs / "staging_report.json"
    return {
        "aois": per_aoi,
        "area_km2": round(sum(v["area_km2"] for v in per_aoi.values()), 1),
        "scenes": sum(sum(v["scenes"].values()) for v in per_aoi.values()),
        "tiles_per_date": sum(v["tiles"] for v in per_aoi.values()),
        "storage_gb": {
            "raw": round(_dir_bytes(paths.raw) / 1e9, 2),
            "models": round(_dir_bytes(paths.models) / 1e9, 2),
            "benchmarks": round(_dir_bytes(paths.benchmarks) / 1e9, 2),
            "disk_free": round(
                shutil.disk_usage(paths.data if paths.data.exists() else paths.root).free / 1e9, 1
            ),
        },
        "latency_ms": LATENCY_MS or None,
        "staging_report": json.loads(report.read_text(encoding="utf-8")) if report.exists() else None,
    }


class TextQuery(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    aoi: str | None = None
    date_range: tuple[str | None, str | None] | None = (
        None  # ISO dates; overrides dates parsed from the query
    )
    sensor: str | None = None  # only sentinel2 composites exist in M2
    k: int = Field(default=T.retrieval.TOP_K_SITES, ge=1, le=200)


@app.post("/search/text")
def search_text(q: TextQuery) -> dict:
    """Parse -> semantic | change | spatial branches -> fusion -> unique sites (M4). Stored as a query for /feedback."""
    from query import parser, retrieve

    t0 = time.perf_counter()
    p = parser.parse(q.query)
    out = retrieve.search(p, retrieve.Request(aoi=q.aoi, date_range=q.date_range, k=q.k))
    qid = db.uuid7()
    with db.engine().begin() as conn:
        conn.execute(
            text(
                "INSERT INTO queries (query_id, text, request, parsed) VALUES (:i, :t, CAST(:r AS jsonb), CAST(:p AS jsonb))"
            ),
            {"i": qid, "t": q.query, "r": q.model_dump_json(), "p": json.dumps(p.as_dict())},
        )
    LATENCY_MS["search_text"] = round((time.perf_counter() - t0) * 1000, 1)
    return {"query_id": str(qid), "parsed": p.as_dict(), **out}


class Target(BaseModel):
    type: str = Field(pattern="^(site|event)$")
    id: str = Field(min_length=1, max_length=80)


class Feedback(BaseModel):
    target: Target
    decision: str = Field(pattern="^(confirm|reject)$")
    reason: str = Field(default="", max_length=500)
    query_id: str | None = None
    actor: str = Field(default="analyst", min_length=1, max_length=60)  # real identity comes with RBAC (M8)


@app.post("/feedback")
def feedback(f: Feedback) -> dict:
    """Store the decision (+ audit, same transaction). Events get analyst_status. With query_id: Rocchio re-rank,
    rejected sites removed (DESIGN.md §3 review loop)."""
    from governance import audit

    with db.engine().begin() as conn:
        if f.target.type == "event":
            n = conn.execute(
                text(
                    "UPDATE change_events SET analyst_status = CAST(:s AS analyst_status) WHERE event_id = :i"
                ),
                {"s": "confirmed" if f.decision == "confirm" else "rejected", "i": f.target.id},
            ).rowcount
            if n == 0:
                raise HTTPException(404, f"unknown event {f.target.id}")
        elif not conn.execute(text("SELECT 1 FROM tiles WHERE tile_id = :t"), {"t": f.target.id}).first():
            raise HTTPException(404, f"unknown site {f.target.id}")
        conn.execute(
            text(
                "INSERT INTO feedback (target_type, target_id, decision, reason, query_id, actor) "
                "VALUES (:tt, :ti, :d, :r, :q, :a)"
            ),
            {
                "tt": f.target.type,
                "ti": f.target.id,
                "d": f.decision,
                "r": f.reason,
                "q": f.query_id,
                "a": f.actor,
            },
        )
        audit.append(conn, f.actor, f"feedback.{f.decision}", f.model_dump())
    return {"recorded": True, **(rerank(f.query_id) if f.query_id else {})}


def rerank(query_id: str) -> dict:
    import numpy as np

    from features import embed_clip
    from query import parser, retrieve
    from query.rocchio import rocchio

    with db.engine().connect() as conn:
        row = conn.execute(
            text("SELECT text, request FROM queries WHERE query_id = CAST(:i AS uuid)"), {"i": query_id}
        ).first()
        if row is None:
            raise HTTPException(404, f"unknown query {query_id}")
        fb = conn.execute(
            text(
                "SELECT DISTINCT ON (target_id) target_id, decision FROM feedback "
                "WHERE query_id = :q AND target_type = 'site' ORDER BY target_id, feedback_id DESC"
            ),
            {"q": query_id},
        ).all()
        decided = {t: d for t, d in fb}
        vecs = (
            {
                t: np.mean([np.asarray(v, dtype=np.float32) for v in vs], axis=0)
                for t, vs in conn.execute(
                    text(
                        "SELECT tile_id, array_agg(text_vec) FROM composites WHERE tile_id = ANY(:t) AND text_vec IS NOT NULL GROUP BY tile_id"
                    ),
                    {"t": list(decided)},
                ).all()
            }
            if decided
            else {}
        )
    req = TextQuery(**row.request)
    p = parser.parse(row.text)
    vec = rocchio(
        embed_clip.embed_text(p.semantic),
        [vecs[t] for t, d in decided.items() if d == "confirm" and t in vecs],
        [vecs[t] for t, d in decided.items() if d == "reject" and t in vecs],
    )
    rejected = frozenset(t for t, d in decided.items() if d == "reject")
    out = retrieve.search(
        p, retrieve.Request(aoi=req.aoi, date_range=req.date_range, k=req.k), vec=vec, exclude_tiles=rejected
    )
    return {"query_id": query_id, "feedback": decided, "sites": out["sites"], "warnings": out["warnings"]}


CHANGE_TYPES = ("construction", "clearance", "water", "road", "other")
EVENT_COLS = (
    "e.event_id, e.tile_id, t.aoi, e.change_type::text AS change_type, e.direction::text AS direction, e.confidence, "
    "e.earliest_supported_date, e.last_normal_date, ST_AsGeoJSON(e.geom) AS geom, e.source_scene_ids, "
    "e.model_versions, e.cascade_version, e.analyst_status::text AS analyst_status"
)


def _event(r) -> dict:
    return {
        "event_id": r["event_id"],
        "tile_id": r["tile_id"],
        "aoi": r["aoi"],
        "change_type": r["change_type"],
        "direction": r["direction"],
        "confidence": r["confidence"],
        "earliest_supported_date": str(r["earliest_supported_date"]),
        "last_normal_date": str(r["last_normal_date"]) if r["last_normal_date"] else None,
        "analyst_status": r["analyst_status"],
        "geometry": json.loads(r["geom"]),
        "provenance": {
            "scene_ids": list(r["source_scene_ids"]),
            "model_versions": r["model_versions"],
            "cascade_version": r["cascade_version"],
        },
    }


@app.get("/change")
def change(
    aoi: str | None = None,
    t1: str | None = None,
    t2: str | None = None,
    type: str | None = None,  # contract parameter name (DESIGN.md §12) shadows the builtin here only
    min_conf: float = T.events.OPERATING_CONF,
) -> dict:
    """Stored events only: confidence is fixed at ingest (§2.2). Below OPERATING_CONF stays hidden unless asked."""
    if type is not None and type not in CHANGE_TYPES:
        raise HTTPException(422, f"type must be one of {CHANGE_TYPES}")
    t0 = time.perf_counter()
    where, params = ["e.confidence >= :c"], {"c": min_conf}
    for cond, key, val in (
        ("t.aoi = :aoi", "aoi", aoi),
        ("e.earliest_supported_date >= CAST(:t1 AS date)", "t1", t1),
        ("e.earliest_supported_date <= CAST(:t2 AS date)", "t2", t2),
        ("e.change_type = CAST(:type AS change_type)", "type", type),
    ):
        if val is not None:
            where.append(cond)
            params[key] = val
    with db.engine().connect() as conn:
        rows = conn.execute(
            text(
                f"SELECT {EVENT_COLS} FROM change_events e JOIN tiles t USING (tile_id) WHERE {' AND '.join(where)} "
                "ORDER BY e.confidence DESC, e.earliest_supported_date"
            ),
            params,
        ).mappings()
        out = [_event(r) for r in rows]
    LATENCY_MS["change"] = round((time.perf_counter() - t0) * 1000, 1)
    return {"events": out}


@app.get("/events/{event_id}")
def event(event_id: str) -> dict:
    with db.engine().connect() as conn:
        r = (
            conn.execute(
                text(
                    f"SELECT {EVENT_COLS}, e.evidence FROM change_events e JOIN tiles t USING (tile_id) "
                    "WHERE e.event_id = :i"
                ),
                {"i": event_id},
            )
            .mappings()
            .first()
        )
    if r is None:
        raise HTTPException(404)
    return _event(r) | {"evidence": r["evidence"]}


@app.get("/chips/{tile_id}/{season}.png")
def chip(tile_id: str, season: str, size: int = 256) -> Response:
    """True-colour chip of one tile's seasonal composite (display only)."""
    aoi = tile_id.split("_")[0]
    f = paths.composites / aoi / f"{season}.tif"
    if aoi not in aois or not f.exists() or not season.replace("-", "").isalnum():
        raise HTTPException(404)
    t = next((t for t in tiling.tiles(aoi, *_grid(aoi)) if t.tile_id == tile_id), None)
    if t is None:
        raise HTTPException(404)
    with rasterio.open(f) as d:
        w = tiling.window(t, d.transform)
        r, g, b = (d.read(d.descriptions.index(n) + 1, window=w) for n in ("red", "green", "blue"))
    buf = io.BytesIO()
    Image.fromarray(rgb_quicklook(r, g, b)).resize((size, size), Image.Resampling.NEAREST).save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png")


def _grid(aoi: str) -> tuple[tuple[float, float, float, float], int]:
    epsg = tiling.aoi_epsg(aois[aoi].bbox)
    return tiling.aoi_bounds_m(aois[aoi].bbox, epsg), epsg


# Workbench (web/, built into dist/ by docker/Dockerfile.api). Mounted last so API routes win.
if (paths.web / "dist").is_dir():
    app.mount("/", StaticFiles(directory=paths.web / "dist", html=True), name="web")
