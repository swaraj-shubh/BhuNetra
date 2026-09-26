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
        for f in [REPO / "index" / "schema.sql", *sorted((REPO / "index" / "migrations").glob("*.sql"))]:
            for stmt in pglast.split(f.read_text(encoding="utf-8")):
                c.exec_driver_sql(stmt)
    eng.dispose()


def _write_scene(folder: Path, date: str, dn: dict, transform, epsg: int, bounds) -> None:
    folder.mkdir(parents=True)
    h, w = dn["red"].shape
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
    layers = {
        name: (dn[canon], "uint16", 0.0001, -0.1) for canon, name in bands["sentinel2"]["bands"].items()
    }
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
    all_tiles = list(tiling.tiles("synth", bounds, epsg))

    def block(rc, margin=0):
        win = tiling.window(next(t for t in all_tiles if (t.row, t.col) == rc), transform)
        r0, c0 = round(win.row_off) + margin, round(win.col_off) + margin
        return slice(r0, r0 + round(win.height) - 2 * margin), slice(c0, c0 + round(win.width) - 2 * margin)

    target = next(t for t in all_tiles if (t.row, t.col) == (1, 1))  # bright structure: the search target
    site = next(t for t in all_tiles if (t.row, t.col) == (3, 3))  # construction appears here in 2024-03
    veg = {
        "blue": 0.04,
        "green": 0.07,
        "red": 0.05,
        "nir": 0.35,
        "nir_narrow": 0.35,
        "swir1": 0.18,
        "swir2": 0.1,
    }
    built = {
        "blue": 0.15,
        "green": 0.18,
        "red": 0.20,
        "nir": 0.25,
        "nir_narrow": 0.25,
        "swir1": 0.35,
        "swir2": 0.3,
    }
    dates = [f"{y}-{m:02d}-10" for y in (2023, 2024) for m in range(1, 13)]  # 24 monthly clear scenes
    for date in dates:
        season = np.cos(2 * np.pi * (int(date[5:7]) - 1) / 12)  # vegetation cycle: NIR peaks in January
        refl = {
            b: np.full((h, w), v + (0.08 * season if b.startswith("nir") else 0.0)) for b, v in veg.items()
        }
        for b, arr in refl.items():
            arr[block((1, 1))] = 0.4
            if date >= "2024-03":
                arr[block((3, 3), margin=17)] = built[b]  # 30 x 30 px construction site
        dn = {
            b: np.round((v + 0.1) / 0.0001).astype(np.uint16) for b, v in refl.items()
        }  # (refl - offset) / scale
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

    assert out["first"]["scene_accepted"] == 24 and out["first"]["season_built"] == 9
    assert out["first"]["postgis_vectors"] == out["first"]["qdrant_points"] > 0
    assert out["again"]["scene_skipped"] == 24 and out["again"]["season_skipped"] == 9  # idempotent
    top = out["top"]
    assert top["tile_id"] == target.tile_id
    assert top["provenance"]["scene_ids"] and all(
        s.startswith("S2X_SYNTH_") for s in top["provenance"]["scene_ids"]
    )
    assert top["provenance"]["model_versions"] == {"text_vec": "fake-clip@0"}
    assert out["ranking_after_reconcile"] == out["ranking"]  # Qdrant rebuilt from PostGIS alone
    assert out["chip_status"] == 200

    # change: ingest -> event exists -> /change returns it (DESIGN.md §11)
    assert out["change_first"]["events"] >= 1 and out["change_again"]["cascade"] == "skipped"
    evs = out["change_events"]
    hit = [e for e in evs if e["change_type"] == "construction" and e["tile_id"] == site.tile_id]
    assert len(hit) == 1, evs
    e = hit[0]
    assert e["earliest_supported_date"] == "2024-03-10" and e["last_normal_date"] == "2024-02-10"
    assert "S2X_SYNTH_2024-03-10" in e["provenance"]["scene_ids"] and e["provenance"]["cascade_version"]
    assert not any(x["tile_id"] == target.tile_id for x in evs)  # a constant bright roof is not change
    detail = out["event_detail"]
    assert len(detail["evidence"]["timeline"]["dates"]) == 24 and detail["evidence"]["calibrated"] is False
    assert out["audit_broken_at"] is None and out["audit_entries"] == 24 + 9 + 1  # scenes + seasons + change
