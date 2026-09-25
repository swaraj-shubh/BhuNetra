"""FastAPI app (DESIGN.md §12). M2: /stats, /search/text (no parser yet: M4), composite chips for the map.

uvicorn api.main:app --port 8000
"""

import io
import json
import shutil
import time
from pathlib import Path

import rasterio
from fastapi import FastAPI, HTTPException, Response
from PIL import Image
from pydantic import BaseModel, Field
from pyproj import Transformer
from qdrant_client import models as qm
from sqlalchemy import text

from configs.settings import T, aois, paths, sources
from index import db, qdrant
from ingest import tiling
from ingest.standardise import rgb_quicklook

LATENCY_MS: dict[str, float] = {}  # last measured, per route (reported by /stats)

app = FastAPI(title="SIH26227 EO archive")


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
    date_range: tuple[str, str] | None = None  # ISO dates; matches composites whose season overlaps it
    sensor: str | None = None  # only sentinel2 composites exist in M2
    k: int = Field(default=T.retrieval.TOP_K_SITES, ge=1, le=200)


@app.post("/search/text")
def search_text(q: TextQuery) -> dict:
    """Encode the query (the only model call on the query path, §2.1) -> Qdrant text_vec -> best season per tile."""
    from features import embed_clip  # torch import only when searching

    t0 = time.perf_counter()
    must: list = []
    if q.aoi:
        must.append(qm.FieldCondition(key="aoi", match=qm.MatchValue(value=q.aoi)))
    if q.date_range:
        must += [
            qm.FieldCondition(key="end_date", range=qm.DatetimeRange(gte=q.date_range[0])),
            qm.FieldCondition(key="start_date", range=qm.DatetimeRange(lte=q.date_range[1])),
        ]
    hits = qdrant.search(embed_clip.embed_text(q.query), "text_vec", T.retrieval.TOP_N, qm.Filter(must=must))
    best: dict[str, qm.ScoredPoint] = {}
    for h in hits:  # hits are score-sorted: keep each tile's best season
        best.setdefault(h.payload["tile_id"], h)
    top = list(best.values())[: q.k]

    with db.engine().connect() as conn:
        rows = conn.execute(
            text(
                "SELECT c.comp_id::text AS comp_id, ST_AsGeoJSON(t.geom) AS geom, c.model_versions, "
                "c.lineage->'observations' AS obs FROM composites c JOIN tiles t USING (tile_id) "
                "WHERE c.comp_id = ANY(CAST(:ids AS uuid[]))"
            ),
            {"ids": [h.payload["comp_id"] for h in top]},
        ).mappings()
        by_comp = {r["comp_id"]: r for r in rows}
    sites = []
    for rank, h in enumerate(top, 1):
        r = by_comp[h.payload["comp_id"]]
        sites.append(
            {
                "rank": rank,
                "tile_id": h.payload["tile_id"],
                "aoi": h.payload["aoi"],
                "season": h.payload["season"],
                "score": round(h.score, 4),
                "q_mean": h.payload["q_mean"],
                "geometry": json.loads(r["geom"]),
                "chip_url": f"/chips/{h.payload['tile_id']}/{h.payload['season']}.png",
                "provenance": {
                    "scene_ids": [o["scene_id"] for o in r["obs"]],
                    "model_versions": r["model_versions"],
                    "cascade_version": None,  # no change cascade involved in a semantic hit
                },
            }
        )
    LATENCY_MS["search_text"] = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "parsed": {"semantic": q.query, "temporal": q.date_range, "spatial": q.aoi, "sensor": q.sensor},
        "sites": sites,
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


def _grid(aoi: str) -> tuple[tuple[float, float, float, float], int]:
    epsg = tiling.aoi_epsg(aois[aoi].bbox)
    return tiling.aoi_bounds_m(aois[aoi].bbox, epsg), epsg
