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
from fastapi import FastAPI, HTTPException, Query, Request, Response
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
WEIGHT_ERROR: list[str] = []  # set at startup when the text model fails verification (M8)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load CLIP and hash its weights once at startup: a cold first query took 26 s (offline run, 2026-09-26)."""
    from features import embed_clip

    try:
        embed_clip.embed_text("warm-up")
        embed_clip.version()
    except FileNotFoundError:  # no weights (CI, laptop without models/): /stats still works, search will 500
        pass
    except Exception as e:  # noqa: BLE001 - e.g. WeightMismatch: keep serving /stats, which reports it
        WEIGHT_ERROR.append(str(e))
    import threading

    threading.Thread(
        target=_warm_cluster, daemon=True
    ).start()  # numba JIT: first "Groups" click took 12-30 s
    yield


def _warm_cluster() -> None:
    import numpy as np

    from query import cluster

    try:  # image upload encodes with Prithvi: load it now, not on the analyst's first upload (~15 s)
        from features import embed_prithvi

        embed_prithvi.model_size()
        embed_prithvi.feature_map(np.zeros((6, 224, 224), np.float32), "2024-01-01", 20.0, 78.0)
    except Exception as e:  # noqa: BLE001 - weights missing (CI/laptop): uploads will report it
        import logging

        logging.getLogger("api").warning("prithvi warm-up skipped: %s", e)

    try:
        cluster.cluster(np.random.default_rng(0).normal(size=(60, 32)).astype(np.float32), 5, 4)
    except Exception as e:  # noqa: BLE001 - warm-up only; a real request reports its own error
        import logging

        logging.getLogger("api").warning("cluster warm-up failed: %s", e)


app = FastAPI(title="SIH26227 EO archive", lifespan=lifespan)

from api.auth import ADMIN, ANALYST, VIEWER
from api.label import router as label_router

app.include_router(label_router)


def _dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def _area_km2(bbox: tuple[float, float, float, float]) -> float:
    epsg = tiling.aoi_epsg(bbox)
    x0, y0, x1, y1 = Transformer.from_crs(4326, epsg, always_xy=True).transform_bounds(*bbox)
    return (x1 - x0) * (y1 - y0) / 1e6


def _index_stats() -> dict | None:
    """Row counts + PostGIS size for the PS report; None when the DB is not reachable (dev laptop)."""
    try:
        with db.engine().connect() as conn:
            r = conn.execute(
                text(
                    "SELECT (SELECT count(*) FROM observations), (SELECT count(*) FROM composites), "
                    "(SELECT count(*) FROM composites WHERE text_vec IS NOT NULL), "
                    "(SELECT count(*) FROM composites WHERE eo_vec IS NOT NULL), (SELECT count(*) FROM change_events), "
                    "pg_database_size(current_database())"
                )
            ).one()
    except Exception:  # noqa: BLE001 - stats must answer without a database
        return None
    return {
        "observations": r[0],
        "composites": r[1],
        "text_vec": r[2],
        "eo_vec": r[3],
        "change_events": r[4],
        "postgis_gb": round(r[5] / 1e9, 2),
    }


@app.get("/stats")
def stats() -> dict:
    """Archive size and footprint. Latencies are not measured yet (M2+)."""
    per_aoi = {}
    for name, a in aois.items():
        epsg = tiling.aoi_epsg(a.bbox)
        scenes = {s: len(list((paths.raw / s / name).glob("*/scene.json"))) for s in sources["sources"]}
        per_aoi[name] = {
            "area_km2": round(_area_km2(a.bbox), 1),
            "bbox": list(a.bbox),
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
        "index": _index_stats(),
        "latency_ms": LATENCY_MS or None,
        "auth": "enabled"
        if __import__("governance.rbac", fromlist=["load"]).load() is not None
        else "disabled (dev)",
        "weights_error": WEIGHT_ERROR or None,
        "staging_report": json.loads(report.read_text(encoding="utf-8")) if report.exists() else None,
        "build": _build_record(),
        "hardware": _hardware(),
        "evaluation": _evaluation(),
    }


def _evaluation() -> dict | None:
    """Latest blind held-out change result (eval/reports/*_m5_heldout_final.json, written by eval/run.py)."""
    runs = sorted((paths.root / "eval" / "reports").glob("*_m5_heldout_final.json"))
    if not runs:
        return None
    ch = json.loads(runs[-1].read_text(encoding="utf-8"))["change"]
    keep = ("n", "precision", "recall", "f1", "false_alarm_rate", "type_accuracy")
    return {
        "file": runs[-1].name,
        "overall": {k: ch["overall"].get(k) for k in keep},
        "per_type": {
            t: {k: v.get(k) for k in ("n", "precision", "recall")} for t, v in ch["per_type"].items()
        },
        "earliest_date_median_err_days": (ch["overall"].get("earliest_date_abs_err_days") or {}).get(
            "median"
        ),
    }


def _build_record() -> dict | None:
    """Seconds per stage of the latest timed full build (manifests/build_full_*.json; PS 2.3 'build time')."""
    runs = sorted((paths.root / "manifests").glob("build_full_*.json"))
    if not runs:
        return None
    from datetime import datetime, timedelta

    t = json.loads(runs[-1].read_text(encoding="utf-8"))["timings"]
    stages = {k: round(sum(e["s"] for e in v)) for k, v in t.items()}
    es = [e for v in t.values() for e in v]
    start = min(datetime.fromisoformat(e["ts"]) - timedelta(seconds=e["s"]) for e in es)
    end = max(datetime.fromisoformat(e["ts"]) for e in es)
    # wall clock: stage timers do not cover every step (e.g. eo_vec batching), so their sum understates the build
    return {"file": runs[-1].name, "wall_s": round((end - start).total_seconds()), "stages_s": stages}


def _hardware() -> dict:
    import os
    import platform
    import subprocess

    import psutil

    gpu = None
    if shutil.which("nvidia-smi"):
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            check=False,
            text=True,
        ).stdout.strip()
    return {
        "cpu": f"{platform.machine()} x{os.cpu_count()}",
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "gpu": gpu or None,  # None: not visible to the api container (the worker holds the GPU)
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
def search_text(q: TextQuery, user: dict = VIEWER) -> dict:
    """Parse -> semantic | change | spatial branches -> fusion -> unique sites (M4). Stored as a query for /feedback."""
    from query import parser, retrieve

    t0 = time.perf_counter()
    p = parser.parse(q.query)
    out = retrieve.search(p, retrieve.Request(aoi=q.aoi, date_range=q.date_range, k=q.k, sensor=q.sensor))
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
def feedback(f: Feedback, user: dict = ANALYST) -> dict:
    """Store the decision (+ audit, same transaction). Events get analyst_status. With query_id: Rocchio re-rank,
    rejected sites removed (DESIGN.md §3 review loop)."""
    from governance import audit

    if user["auth"]:  # authenticated identity wins over the self-declared actor
        f = f.model_copy(update={"actor": user["name"]})
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
        p,
        retrieve.Request(aoi=req.aoi, date_range=req.date_range, k=req.k, sensor=req.sensor),
        vec=vec,
        exclude_tiles=rejected,
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
    unreviewed: bool = False,  # workbench queue: only events no analyst has decided yet
    all_types: bool = False,  # include events.HIDDEN_TYPES (hidden by default: precision first, §2.11)
    bbox: str | None = None,  # "min_lon,min_lat,max_lon,max_lat": any analyst-specified area (PS 2.2.2)
    limit: int | None = Query(
        default=None, ge=1, le=100000
    ),  # workbench page size; evaluation omits it (all events)
    user: dict = VIEWER,
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
    if unreviewed:
        where.append("e.analyst_status = 'unreviewed'")
    if bbox is not None:
        try:
            x0, y0, x1, y1 = (float(v) for v in bbox.split(","))
        except ValueError:
            raise HTTPException(422, "bbox must be min_lon,min_lat,max_lon,max_lat") from None
        where.append("ST_Intersects(e.geom, ST_MakeEnvelope(:bx0, :by0, :bx1, :by1, 4326))")
        params |= {"bx0": x0, "by0": y0, "bx1": x1, "by1": y1}
    if type is None and not all_types and T.events.HIDDEN_TYPES:
        where.append("NOT (e.change_type::text = ANY(:hidden))")
        params["hidden"] = list(T.events.HIDDEN_TYPES)
    cond_sql = " AND ".join(where)
    with db.engine().connect() as conn:
        total = conn.execute(
            text(f"SELECT count(*) FROM change_events e JOIN tiles t USING (tile_id) WHERE {cond_sql}"),
            params,
        ).scalar_one()
        rows = conn.execute(
            text(
                f"SELECT {EVENT_COLS} FROM change_events e JOIN tiles t USING (tile_id) WHERE {cond_sql} "
                "ORDER BY e.confidence DESC, e.earliest_supported_date, e.event_id"
                + (" LIMIT :lim" if limit else "")
            ),
            params | ({"lim": limit} if limit else {}),
        ).mappings()
        out = [_event(r) for r in rows]
    LATENCY_MS["change"] = round((time.perf_counter() - t0) * 1000, 1)
    return {"events": out, "total": total}


@app.get("/events/{event_id}")
def event(event_id: str, user: dict = VIEWER) -> dict:
    with db.engine().connect() as conn:
        r = (
            conn.execute(
                text(
                    f"SELECT {EVENT_COLS}, e.evidence, e.sensors FROM change_events e JOIN tiles t USING (tile_id) "
                    "WHERE e.event_id = :i"
                ),
                {"i": event_id},
            )
            .mappings()
            .first()
        )
        if r is None:
            raise HTTPException(404)
        extra = _event_context(conn, r)
    return _event(r) | {"evidence": r["evidence"]} | extra


def _pick_images(comps: list[dict], ln: str | None, es: str) -> dict:
    """Before = latest composite ending on/before the last normal date; after = first composite starting on/after
    the earliest supported date, same season of year as before if possible. Dry seasons preferred (monsoon patchy)."""
    wet = lambda c: c["season"].endswith("JJAS")
    pre = [c for c in comps if c["end_date"] <= (ln or es)]
    before = max(pre, key=lambda c: (not wet(c), c["end_date"]), default=None)
    post = [c for c in comps if c["start_date"] >= es]
    same = [c for c in post if before and c["season"].split("-")[1] == before["season"].split("-")[1]]
    after = min(same or post, key=lambda c: (wet(c), c["start_date"]), default=None)
    return {"before": before, "after": after}


def _event_context(conn, r) -> dict:
    """What an analyst needs next to the event (PS 2.2.5): location, size, the actual observations and sensors behind
    the dates, the composites shown as before/after (with their acquisition dates), and the processing history."""
    loc = conn.execute(
        text(
            "SELECT ST_Y(ST_Centroid(e.geom)), ST_X(ST_Centroid(e.geom)), ST_Area(e.geom::geography) / 1e4, "
            "ST_AsGeoJSON(t.geom) FROM change_events e JOIN tiles t USING (tile_id) WHERE e.event_id = :i"
        ),
        {"i": r["event_id"]},
    ).one()
    dates = [d for d in (r["last_normal_date"], r["earliest_supported_date"]) if d]
    obs = {
        str(o["acq_date"]): dict(o)
        for o in conn.execute(
            text(
                "SELECT s.acq_date, s.scene_id, s.source, round(o.q::numeric, 2) AS q FROM observations o "
                "JOIN scenes s USING (scene_id) WHERE o.tile_id = :t AND s.acq_date = ANY(:d)"
            ),
            {"t": r["tile_id"], "d": dates},
        ).mappings()
    }
    comps = [
        {
            "season": c["season"],
            "start_date": str(c["start_date"]),
            "end_date": str(c["end_date"]),
            "n_obs": c["n_obs"],
            "dates": sorted(o["date"] for o in c["obs"] or []),
            "scene_ids": sorted(o["scene_id"] for o in c["obs"] or []),
            "chip_url": f"/chips/{r['tile_id']}/{c['season']}.png",
        }
        for c in conn.execute(
            text(
                "SELECT season, start_date, end_date, n_obs, lineage->'observations' AS obs FROM composites "
                "WHERE tile_id = :t ORDER BY start_date"
            ),
            {"t": r["tile_id"]},
        ).mappings()
    ]
    ln = str(r["last_normal_date"]) if r["last_normal_date"] else None
    history = [
        {
            "ts": h["ts"].isoformat(timespec="seconds"),
            "actor": h["actor"],
            "action": h["action"],
            "payload": h["payload"],
        }
        for h in conn.execute(
            text(
                "SELECT ts, actor, action, payload FROM audit_log WHERE "
                "(action IN ('change.run', 'change.sar_evidence', 'change.landsat_evidence', 'change.calibrate') "
                "AND payload->>'aoi' = :a) OR (action LIKE 'feedback.%' AND payload->'target'->>'id' = :i) "
                "OR (action = 'export' AND payload::text LIKE :p) ORDER BY seq"
            ),
            {"a": r["aoi"], "i": r["event_id"], "p": f'%"{r["event_id"]}"%'},
        ).mappings()
    ]
    runs = [i for i, h in enumerate(history) if h["action"] == "change.run"]
    history = (
        history[runs[-1] :] if runs else history
    )  # event ids are new on every run: earlier runs made other events
    return {
        "location": {"lat": round(loc[0], 5), "lon": round(loc[1], 5)},
        "tile_geometry": json.loads(loc[3]),  # footprint of the before/after images (map context)
        "area_ha": round(loc[2], 2),
        "sensors": list(r["sensors"]),
        "observations": {
            "last_normal": obs.get(ln) if ln else None,
            "first_changed": obs.get(str(r["earliest_supported_date"])),
        },
        "images": _pick_images(comps, ln, str(r["earliest_supported_date"])),
        "history": history,
    }


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


@app.get("/chips/{tile_id}/{season}.tif")
def chip_tif(tile_id: str, season: str) -> Response:
    """The tile's 6-band reflectance composite as a GeoTIFF (for image search by upload, and for analysts)."""
    import numpy as np
    from rasterio.io import MemoryFile

    aoi = tile_id.split("_")[0]
    f = paths.composites / aoi / f"{season}.tif"
    if aoi not in aois or not f.exists() or not season.replace("-", "").isalnum():
        raise HTTPException(404)
    t = next((t for t in tiling.tiles(aoi, *_grid(aoi)) if t.tile_id == tile_id), None)
    if t is None:
        raise HTTPException(404)
    with rasterio.open(f) as d:
        w = tiling.window(t, d.transform)
        names = [n for n in d.descriptions if n]
        arr = np.stack([d.read(d.descriptions.index(n) + 1, window=w) for n in names]).astype(np.float32)
        prof = {
            "driver": "GTiff",
            "dtype": "float32",
            "count": len(names),
            "height": arr.shape[1],
            "width": arr.shape[2],
            "crs": d.crs,
            "transform": d.window_transform(w),
            "nodata": float("nan"),
        }
    with MemoryFile() as mem:
        with mem.open(**prof) as out:
            out.write(arr)
            for i, n in enumerate(names, 1):
                out.set_band_description(i, n)
            out.update_tags(season=season, tile_id=tile_id, source="EO Archive composite (reflectance 0-1)")
        data = mem.read()
    return Response(
        data,
        media_type="image/tiff",
        headers={"Content-Disposition": f'attachment; filename="{tile_id}_{season}.tif"'},
    )


def _embed_chip(b64: str, date: str | None) -> list[float]:
    """Uploaded multiband GeoTIFF -> eo_vec (Prithvi), the same encoder as the archive. Encoding the query chip is
    the one model call the query path allows (DESIGN.md §2.1). Bands are found by description (blue, green, red,
    nir or nir_narrow, swir1, swir2); digital numbers 0-10000 are scaled to reflectance."""
    import base64
    import binascii
    import tempfile
    from datetime import UTC, datetime

    import numpy as np

    from features import embed_prithvi

    try:
        raw = base64.b64decode(b64, validate=True)
    except binascii.Error:
        raise HTTPException(422, "chip must be a base64-encoded GeoTIFF") from None
    if len(raw) > 30_000_000:
        raise HTTPException(413, "chip larger than 30 MB")
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "chip.tif"
        src.write_bytes(raw)
        try:
            with rasterio.open(src) as ds:
                names = [(n or "").lower() for n in ds.descriptions]
                used = embed_prithvi.bands_used(names)
                x = np.stack([ds.read(names.index(b) + 1).astype(np.float32) for b in used])
                h, w, crs, bounds = ds.height, ds.width, ds.crs, ds.bounds
        except rasterio.errors.RasterioIOError:
            raise HTTPException(422, "chip is not a readable GeoTIFF") from None
        except ValueError as e:
            raise HTTPException(422, f"{e}; name the bands blue, green, red, nir, swir1, swir2") from None
    if max(h, w) > 1024 or min(h, w) < 16:
        raise HTTPException(422, "chip must be 16-1024 px on each side")
    if np.nanmax(x) > 2.0:  # digital numbers (e.g. Sentinel-2 L2A 0-10000) -> reflectance
        x = x / 10000.0
    lon, lat = Transformer.from_crs(crs, 4326, always_xy=True).transform(
        (bounds.left + bounds.right) / 2, (bounds.top + bounds.bottom) / 2
    )
    # Archive vectors pool a tile out of a 224 px context window (DESIGN.md §7). A small chip has no context, so it is
    # reflect-padded to the model size: cosine to the stored vector of the same tile 0.913 unpadded -> 0.977 padded
    # (2026-09-28, hasdeo_44N_0030_0031 2024-MAM; full real context 0.996).
    size = embed_prithvi.model_size()
    py, px = max(size - h, 0) // 2, max(size - w, 0) // 2
    if py or px:
        x = np.pad(x, ((0, 0), (py, max(size - h, 0) - py), (px, max(size - w, 0) - px)), mode="reflect")
    fmap = embed_prithvi.feature_map(x, date or datetime.now(UTC).date().isoformat(), lat, lon)
    return embed_prithvi.pool_tiles(fmap, [(py, px)], tile_px=min(h, w))[0].tolist()


def _grid(aoi: str) -> tuple[tuple[float, float, float, float], int]:
    epsg = tiling.aoi_epsg(aois[aoi].bbox)
    return tiling.aoi_bounds_m(aois[aoi].bbox, epsg), epsg


# Workbench (web/, built into dist/ by docker/Dockerfile.api). Mounted last so API routes win.


class ImageQuery(BaseModel):
    tile_id: str | None = None  # a site already in the archive (its stored eo_vec)
    season: str | None = None  # which season of that tile; default: its latest with an eo_vec
    chip: str | None = (
        None  # base64 multiband GeoTIFF (bands named blue..swir2), encoded by Prithvi at query time
    )
    chip_date: str | None = None  # acquisition date of the chip (Prithvi's temporal input); default today
    aoi: str | None = None
    date_range: tuple[str | None, str | None] | None = None
    k: int = Field(default=T.retrieval.TOP_K_SITES, ge=1, le=200)


@app.post("/search/image")
def search_image(q: ImageQuery, user: dict = VIEWER) -> dict:
    """Image -> image via eo_vec (Prithvi), never text_vec (DESIGN.md §2.3). The seed site itself is excluded."""
    import numpy as np

    from query import retrieve

    t0 = time.perf_counter()
    if q.chip is not None:  # uploaded image: same encoder, same index, nothing excluded
        vec = np.asarray(_embed_chip(q.chip, q.chip_date), np.float32)
        sites = retrieve.search_image(vec, q.aoi, q.date_range, q.k)
        LATENCY_MS["search_image_upload"] = round((time.perf_counter() - t0) * 1000, 1)
        return {"seed": {"tile_id": None, "season": None, "upload": True}, "sites": sites}
    if not q.tile_id:
        raise HTTPException(422, "give tile_id or chip")
    with db.engine().connect() as conn:
        seed = retrieve.eo_vector(conn, q.tile_id, q.season)
    if seed is None:
        raise HTTPException(404, f"no eo_vec for {q.tile_id} (run make embed-eo)")
    vec, season = seed
    sites = retrieve.search_image(
        np.asarray(vec, np.float32), q.aoi, q.date_range, q.k, retrieve.overlapping(q.tile_id)
    )
    LATENCY_MS["search_image"] = round((time.perf_counter() - t0) * 1000, 1)
    return {"seed": {"tile_id": q.tile_id, "season": season}, "sites": sites}


class ClusterRequest(BaseModel):
    seed_tile_id: str | None = None
    aoi: str | None = None
    season: str | None = None
    n: int = Field(default=T.retrieval.TOP_N, ge=20, le=5000)  # seed mode: how many look-alikes to group


@app.post("/discover/cluster")
def discover_cluster(r: ClusterRequest, user: dict = VIEWER) -> dict:
    """Look-alike groups (HDBSCAN on UMAP of centred eo_vec). Seed: its n nearest sites; AOI: one season, all tiles."""
    import numpy as np

    from query import cluster, retrieve

    t0 = time.perf_counter()
    with db.engine().connect() as conn:
        if r.seed_tile_id:
            seed = retrieve.eo_vector(conn, r.seed_tile_id, r.season)
            if seed is None:
                raise HTTPException(404, f"no eo_vec for {r.seed_tile_id}")
            hits = retrieve.semantic(np.asarray(seed[0], np.float32), r.aoi, None, using="eo_vec")
            keys = [(h.payload or {})["tile_id"] for h in list(hits.values())[: r.n]]
            rows = conn.execute(
                text(
                    "SELECT DISTINCT ON (c.tile_id) c.tile_id, c.season, c.eo_vec, ST_AsGeoJSON(t.geom) FROM composites c "
                    "JOIN tiles t USING (tile_id) WHERE c.tile_id = ANY(:t) AND c.eo_vec IS NOT NULL "
                    "ORDER BY c.tile_id, c.end_date DESC"
                ),
                {"t": keys + [r.seed_tile_id]},
            ).all()
        elif r.aoi and r.season:
            rows = conn.execute(
                text(
                    "SELECT c.tile_id, c.season, c.eo_vec, ST_AsGeoJSON(t.geom) FROM composites c JOIN tiles t "
                    "USING (tile_id) WHERE t.aoi = :a AND c.season = :s AND c.eo_vec IS NOT NULL"
                ),
                {"a": r.aoi, "s": r.season},
            ).all()
        else:
            raise HTTPException(422, "give seed_tile_id, or aoi + season")
    if len(rows) < 3:
        return {"clusters": [], "noise": [row[0] for row in rows]}
    vecs = np.asarray([row[2] for row in rows], np.float32)
    labels = cluster.cluster(vecs, T.cluster.MIN_CLUSTER_SIZE, T.cluster.UMAP_DIM)
    out = []
    for lab in sorted(set(labels) - {-1}):
        idx = np.flatnonzero(labels == lab)
        centre = vecs[idx].mean(0)
        medoid = int(idx[np.argmax(vecs[idx] @ centre)])
        out.append(
            {
                "cluster": int(lab),
                "size": len(idx),
                "medoid_tile": rows[medoid][0],
                "contains_seed": bool(r.seed_tile_id and r.seed_tile_id in {rows[i][0] for i in idx}),
                "members": [
                    {"tile_id": rows[i][0], "season": rows[i][1], "geometry": json.loads(rows[i][3])}
                    for i in idx
                ],
            }
        )
    out.sort(key=lambda c: (-c["contains_seed"], -c["size"]))
    LATENCY_MS["discover_cluster"] = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "clusters": out,
        "noise": [rows[i][0] for i in np.flatnonzero(labels == -1)],
        "n_points": len(rows),
    }


class IngestRequest(BaseModel):
    path: str = Field(
        min_length=1, max_length=500
    )  # staged folder or GeoTIFF/COG under data/ (worker validates)
    aoi: str


@app.post("/ingest")
def ingest_job(req: IngestRequest, user: dict = ADMIN) -> dict:
    """Queue an incremental ingest (M9). The worker service does the work (DESIGN.md §2.1)."""
    from governance import audit

    if req.aoi not in aois:
        raise HTTPException(422, f"unknown AOI {req.aoi}")
    with db.engine().begin() as conn:
        job_id = conn.execute(
            text(
                "INSERT INTO jobs (kind, params) VALUES ('ingest_scene', CAST(:p AS jsonb)) RETURNING job_id"
            ),
            {"p": req.model_dump_json()},
        ).scalar()
        audit.append(conn, user["name"], "ingest.queued", {"job_id": str(job_id), **req.model_dump()})
    return {"job_id": str(job_id)}


@app.get("/jobs/{job_id}")
def job(job_id: str, user: dict = VIEWER) -> dict:
    with db.engine().connect() as conn:
        r = (
            conn.execute(
                text(
                    "SELECT job_id::text, kind, params, status, timings, error, created_ts, finished_ts FROM jobs "
                    "WHERE job_id::text = :i"
                ),
                {"i": job_id},
            )
            .mappings()
            .first()
        )
    if r is None:
        raise HTTPException(404)
    return {k: (str(v) if k.endswith("_ts") and v else v) for k, v in r.items()}


class ExportRequest(BaseModel):
    event_ids: list[str] | None = Field(default=None, max_length=5000)
    query_id: str | None = None
    format: str = Field(default="geojson", pattern="^(geojson|gpkg)$")


def _export_key():
    from governance import sign

    return sign.load_or_create_key(paths.data / "keys" / "export_ed25519.pem")


@app.post("/export")
def export(req: ExportRequest, user: dict = ANALYST) -> Response:
    """Ed25519-signed zip: events.geojson (+ sites.geojson for a query) [+ export.gpkg] + provenance + manifest (§3)."""
    import zipfile

    from governance import audit, sign

    if not (req.event_ids or req.query_id):
        raise HTTPException(422, "give event_ids or query_id")
    files: dict[str, bytes] = {}
    ids = list(req.event_ids or [])
    if req.query_id:
        sites = rerank(req.query_id)["sites"]
        files["sites.geojson"] = json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": s["geometry"],
                        "properties": {k: v for k, v in s.items() if k != "geometry"},
                    }
                    for s in sites
                ],
            }
        ).encode()
        ids += [e["event_id"] for s in sites for e in s["events"]]
    with db.engine().connect() as conn:
        rows = (
            conn.execute(
                text(
                    f"SELECT {EVENT_COLS}, e.evidence FROM change_events e JOIN tiles t USING (tile_id) WHERE e.event_id = ANY(:i)"
                ),
                {"i": sorted(set(ids))},
            )
            .mappings()
            .all()
        )
    evs = [_event(r) | {"evidence": r["evidence"]} for r in rows]
    files["events.geojson"] = json.dumps(
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": e["geometry"],
                    "properties": {k: v for k, v in e.items() if k not in ("geometry", "evidence")},
                }
                for e in evs
            ],
        }
    ).encode()
    if req.format == "gpkg":  # same features as the GeoJSON, one layer each, inside the same signed package
        from api import gpkg

        files["export.gpkg"] = gpkg.write(
            {name[: -len(".geojson")]: json.loads(files[name])["features"] for name in sorted(files)}
        )
    files["provenance.json"] = json.dumps(
        {e["event_id"]: {"provenance": e["provenance"], "evidence": e["evidence"]} for e in evs}
    ).encode()
    export_id = str(db.uuid7())
    pkg = sign.build_package(
        files,
        _export_key(),
        meta={"export_id": export_id, "actor": user["name"], "query_id": req.query_id, "n_events": len(evs)},
    )
    with db.engine().begin() as conn:
        audit.append(
            conn,
            user["name"],
            "export",
            {
                "export_id": export_id,
                "manifest_sha256": sign.sha256_hex(pkg["manifest.json"]),
                "n_events": len(evs),
                "query_id": req.query_id,
            },
        )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, b in pkg.items():
            z.writestr(name, b)
    return Response(
        buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="export_{export_id}.zip"'},
    )


@app.post("/export/verify")
async def export_verify(request: Request, user: dict = VIEWER) -> dict:
    """Upload a zip (raw body); verified against THIS server's export key (origin + integrity)."""
    import zipfile

    from governance import sign

    try:
        with zipfile.ZipFile(io.BytesIO(await request.body())) as z:
            pkg = {n: z.read(n) for n in z.namelist()}
    except zipfile.BadZipFile as e:
        raise HTTPException(422, "not a zip") from e
    return sign.verify_package(pkg, trusted_public_pem=sign.public_pem(_export_key()))


@app.get("/audit/verify")
def audit_verify(user: dict = VIEWER) -> dict:
    from governance import audit

    with db.engine().connect() as conn:
        entries = audit.load(conn)
    broken = audit.verify(entries)
    return {"ok": broken is None, "entries": len(entries), "broken_at": broken}


if (paths.web / "dist").is_dir():
    app.mount("/", StaticFiles(directory=paths.web / "dist", html=True), name="web")
