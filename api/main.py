"""FastAPI app (DESIGN.md §12). M0: /stats. The other contract routes land with their milestones.

uvicorn api.main:app --port 8000
"""

import json
import shutil
from pathlib import Path

from fastapi import FastAPI
from pyproj import Transformer

from configs.settings import aois, paths, sources
from ingest import tiling

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
        "latency_ms": None,  # TODO(measure) once /search exists
        "staging_report": json.loads(report.read_text(encoding="utf-8")) if report.exists() else None,
    }
