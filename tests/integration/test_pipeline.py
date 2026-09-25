"""End-to-end on a synthetic AOI (DESIGN.md §11): ingest -> composite -> vectors -> PostGIS/Qdrant -> /search/text.

Needs TEST_DATABASE_URL pointing at a THROWAWAY PostGIS database whose name contains "test" (the schema is dropped and
re-created). Skipped otherwise. CI provides one (.github/workflows/ci.yml); on the AWS box:
    docker compose ... exec db createdb -U sih sih_test
    TEST_DATABASE_URL=postgresql://sih:<pw>@127.0.0.1:5432/sih_test uv run pytest tests/integration -q
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pglast
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin
from sqlalchemy import create_engine

from configs.settings import bands
from ingest import tiling
from ingest.validate import sha256

REPO = Path(__file__).resolve().parents[2]
URL = os.environ.get("TEST_DATABASE_URL", "")
BBOX = (72.20, 22.25, 72.22, 22.27)  # ~2 km: a few tiles each way
pytestmark = pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")


def _reset_db() -> None:
    if "test" not in URL.rsplit("/", 1)[-1]:
        raise RuntimeError("TEST_DATABASE_URL must name a throwaway database containing 'test'")
    eng = create_engine(URL.replace("postgresql://", "postgresql+pg8000://", 1))
    with eng.begin() as c:
        c.exec_driver_sql("DROP SCHEMA IF EXISTS public CASCADE")
        c.exec_driver_sql("CREATE SCHEMA public")
        for stmt in pglast.split((REPO / "index" / "schema.sql").read_text(encoding="utf-8")):
            c.exec_driver_sql(stmt)
    eng.dispose()


def _write_scene(folder: Path, date: str, dn: np.ndarray, transform, epsg: int, bounds) -> None:
    folder.mkdir(parents=True)
    h, w = dn.shape
    prof = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "crs": f"EPSG:{epsg}",
        "transform": transform,
    }
    meta = {
        "aoi": "synth",
        "sensor": "sentinel2",
        "date": date,
        "datetime": f"{date}T05:30:00Z",
        "epsg": epsg,
        "bounds": list(bounds),
        "source_items": [f"S2X_SYNTH_{date}"],
        "properties": {"view:sun_elevation": 50.0},
        "bands": {},
    }
    layers = {name: (dn, "uint16", 0.0001, -0.1) for name in bands["sentinel2"]["bands"].values()}
    layers["SCL"] = (np.full((h, w), 4, np.uint8), "uint8", 1.0, 0.0)  # all clear vegetation
    layers["CLD"] = (np.zeros((h, w), np.uint8), "uint8", 1.0, 0.0)
    for name, (arr, dtype, scale, offset) in layers.items():
        with rasterio.open(folder / f"{name}.tif", "w", dtype=dtype, nodata=0, **prof) as d:
            d.write(arr, 1)
        meta["bands"][name] = {
            "file": f"{name}.tif",
            "sha256": sha256(folder / f"{name}.tif"),
            "scale": scale,
            "offset": offset,
            "nodata": 0.0,
        }
    (folder / "scene.json").write_text(json.dumps(meta), encoding="utf-8")


def test_synthetic_aoi_end_to_end(tmp_path):
    _reset_db()
    root = tmp_path / "root"
    shutil.copytree(REPO / "configs", root / "configs", ignore=shutil.ignore_patterns("__pycache__"))
    for f in (root / "configs" / "aoi").glob("*.yaml"):
        f.unlink()
    (root / "configs" / "aoi" / "synth.yaml").write_text(
        yaml.safe_dump({"name": "synth", "type": "test", "why": "integration test", "bbox": list(BBOX)}),
        encoding="utf-8",
    )
    (root / "manifests").mkdir()

    epsg = tiling.aoi_epsg(BBOX)
    bounds = tiling.aoi_bounds_m(BBOX, epsg)
    gsd = 10.0
    transform = from_origin(bounds[0], bounds[3], gsd, gsd)
    h, w = round((bounds[3] - bounds[1]) / gsd), round((bounds[2] - bounds[0]) / gsd)
    dn = np.full((h, w), 1500, np.uint16)  # dark background (reflectance 0.05)
    target = next(t for t in tiling.tiles("synth", bounds, epsg) if (t.row, t.col) == (1, 1))
    win = tiling.window(target, transform)
    r0, c0 = round(win.row_off), round(win.col_off)
    dn[r0 : r0 + round(win.height), c0 : c0 + round(win.width)] = 5000  # bright "structure" filling one tile
    for date in ("2024-01-10", "2024-02-05"):  # both in 2024-DJF
        _write_scene(root / "data" / "raw" / "sentinel2" / "synth" / date, date, dn, transform, epsg, bounds)

    env = {k: v for k, v in os.environ.items() if k not in ("QDRANT_URL", "DATABASE_URL")}
    env |= {
        "SIH_ROOT": str(root),
        "DATABASE_URL": URL,
        "PYTHONPATH": str(REPO),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    run = subprocess.run(  # noqa: PLW1510 - returncode asserted below with stderr
        [sys.executable, str(REPO / "tests" / "integration" / "driver.py")],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
        cwd=REPO,
    )
    assert run.returncode == 0, run.stderr[-3000:]
    out = json.loads(run.stdout.strip().splitlines()[-1])

    assert out["first"]["scene_accepted"] == 2 and out["first"]["season_built"] == 1
    assert out["first"]["postgis_vectors"] == out["first"]["qdrant_points"] > 0
    assert out["again"]["scene_skipped"] == 2 and out["again"]["season_skipped"] == 1  # idempotent
    top = out["top"]
    assert top["tile_id"] == target.tile_id and top["season"] == "2024-DJF"
    assert top["provenance"]["scene_ids"] == ["S2X_SYNTH_2024-01-10", "S2X_SYNTH_2024-02-05"]
    assert top["provenance"]["model_versions"] == {"text_vec": "fake-clip@0"}
    assert out["ranking_after_reconcile"] == out["ranking"]  # Qdrant rebuilt from PostGIS alone
    assert out["audit_broken_at"] is None and out["audit_entries"] == 3  # 2 scenes + 1 composite
    assert out["chip_status"] == 200
