"""Change cascade for one AOI (M3): ARD indices -> per-pixel breaks -> events -> PostGIS (+ audit, same transaction).

    python -m worker.change <aoi> [<aoi> ...]      (make change AOI=...)

Idempotent: ARD per scene is skipped when its scene fingerprint is unchanged; the cascade is skipped when its inputs,
CASCADE_VERSION and the change thresholds are unchanged. A re-run replaces this AOI's *unreviewed* events (a
versioned re-run, DESIGN.md §2.2); analyst-reviewed events are kept.
"""

import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import rasterio
from sqlalchemy import text

from change import cascade, events
from configs.settings import PROCESSING_VERSION, T, aois, paths
from features import composites, indices
from governance import audit
from index import db
from ingest import quality, scene, tiling
from worker.ingest import ACTOR, timed

log = logging.getLogger("worker.change")
INDEX_BANDS = {"blue", "green", "red", "nir", "swir1"}  # features/indices.py needs all of them


def ensure_ard(aoi: str, tiles: list[tiling.Tile]) -> int:
    """Index rasters for every accepted scene of the AOI. Returns how many were (re)built."""
    with db.engine().connect() as conn:
        folders = [
            paths.root / r[0]
            for r in conn.execute(
                text("SELECT raw_path FROM scenes WHERE aoi = :a AND status = 'accepted' ORDER BY acq_date"),
                {"a": aoi},
            )
        ]
    built = 0
    for f in folders:
        out = paths.data / "ard" / aoi / f"{f.name}.tif"
        fp = scene.fingerprint(f)
        side = out.with_suffix(".json")
        if (
            out.exists()
            and side.exists()
            and json.loads(side.read_text(encoding="utf-8"))["fingerprint"] == fp
        ):
            continue
        obs = scene.load(f)
        if missing := INDEX_BANDS - set(obs.refl):
            log.info(json.dumps({"scene": f.name, "ard": "skipped", "missing_bands": sorted(missing)}))
            continue
        vf, cn = quality.tile_fractions(obs.valid, obs.cloud, obs.transform, tiles)
        q = quality.q_score(vf, np.nan, cn, obs.sun_elev if obs.sun_elev is not None else np.nan)
        usable = obs.valid & composites.q_ok_mask(q, tiles, obs.transform, obs.valid.shape)
        indices.write(
            out,
            indices.compute(obs.refl),
            usable,
            obs.transform,
            obs.epsg,
            {
                "scene_id": obs.scene_id,
                "fingerprint": fp,
                "processing_version": PROCESSING_VERSION,
                "Q_MIN": T.quality.Q_MIN,
            },
        )
        built += 1
    return built


def _state_key(ard_dir: Path) -> str:
    fps = [
        json.loads(j.read_text(encoding="utf-8"))["fingerprint"]
        for j in sorted(ard_dir.glob("????-??-??.json"))
    ]
    cfg = T.model_dump(include={"quality", "breaks", "classify", "events", "composites"})
    return hashlib.sha256(json.dumps([fps, events.CASCADE_VERSION, cfg], sort_keys=True).encode()).hexdigest()


def _tile_of(tiles: list[tiling.Tile], x: float, y: float) -> str:
    centres = np.array(
        [((t.bounds_m[0] + t.bounds_m[2]) / 2, (t.bounds_m[1] + t.bounds_m[3]) / 2) for t in tiles]
    )
    return tiles[int(np.argmin(((centres - (x, y)) ** 2).sum(1)))].tile_id


def run(aoi: str) -> dict:
    a = aois[aoi]
    epsg = tiling.aoi_epsg(a.bbox)
    tiles = list(tiling.tiles(aoi, tiling.aoi_bounds_m(a.bbox, epsg), epsg))
    with timed("ard", aoi=aoi):
        n_ard = ensure_ard(aoi, tiles)
    ard_dir = paths.data / "ard" / aoi
    out_dir = paths.data / "change" / aoi
    if len(list(ard_dir.glob("????-??-??.tif"))) < T.breaks.MIN_BASELINE_OBS + T.breaks.K_PERSIST:
        # e.g. RGB-only organiser imagery: semantic search works, index-based change does not (§16)
        return {
            "aoi": aoi,
            "ard_built": n_ard,
            "cascade": "unavailable",
            "reason": f"needs >= {T.breaks.MIN_BASELINE_OBS + T.breaks.K_PERSIST} dates with {sorted(INDEX_BANDS)}",
        }
    key = _state_key(ard_dir)
    state_f = out_dir / "state.json"
    if state_f.exists() and json.loads(state_f.read_text(encoding="utf-8")).get("key") == key:
        return {"aoi": aoi, "ard_built": n_ard, "cascade": "skipped"}

    pix_key = (
        out_dir / "pixels.json"
    )  # the per-pixel pass is the slow part (~15 min/AOI): reuse it when inputs match
    if (
        (out_dir / "pixels.tif").exists()
        and pix_key.exists()
        and json.loads(pix_key.read_text(encoding="utf-8")) == key
    ):
        s = cascade.series(ard_dir)
    else:
        with timed("cascade", aoi=aoi):
            s = cascade.run(ard_dir, out_dir / "pixels.tif")
        pix_key.write_text(json.dumps(key), encoding="utf-8")
    with rasterio.open(out_dir / "pixels.tif") as d:
        pix = {b: d.read(i + 1) for i, b in enumerate(cascade.BANDS)}
        transform = d.transform
    with timed("events", aoi=aoi):
        evs = events.extract(pix, s.dates, transform, epsg)
        scene_ids = [
            json.loads(f.with_suffix(".json").read_text(encoding="utf-8"))["scene_id"] for f in s.files
        ]
        tls = events.timelines([e.pixels for e in evs], (indices.read(f) for f in s.files))

    k = T.breaks.K_PERSIST
    rows = []
    for e, tl in zip(evs, tls, strict=True):
        used = sorted(
            {i for i in [e.last_normal_idx, *range(e.break_idx, e.break_idx + k)] if 0 <= i < len(s.dates)}
        )
        rows.append(
            {
                "tile": _tile_of(tiles, *e.centroid_m),
                "wkt": e.geom_wkt,
                "ct": e.change_type,
                "dir": e.direction,
                "conf": e.confidence,
                "ln": s.dates[e.last_normal_idx] if e.last_normal_idx >= 0 else None,
                "es": s.dates[e.break_idx],
                "scenes": [scene_ids[i] for i in used],
                "ev": json.dumps(e.evidence | {"timeline": {"dates": s.dates, **tl}}),
                "mv": json.dumps({"cascade": events.CASCADE_VERSION}),
            }
        )
    by_type: dict[str, int] = {}
    for r in rows:
        by_type[r["ct"]] = by_type.get(r["ct"], 0) + 1
    with timed("events_commit", aoi=aoi, n=len(rows)), db.engine().begin() as conn:
        conn.execute(
            text(
                "DELETE FROM change_events WHERE analyst_status = 'unreviewed' "
                "AND tile_id IN (SELECT tile_id FROM tiles WHERE aoi = :a)"
            ),
            {"a": aoi},
        )
        if rows:
            conn.execute(
                text(
                    "INSERT INTO change_events (tile_id, geom, change_type, direction, confidence, last_normal_date, "
                    "earliest_supported_date, sensors, source_scene_ids, evidence, model_versions, cascade_version) "
                    "VALUES (:tile, ST_GeomFromText(:wkt, 4326), CAST(:ct AS change_type), CAST(:dir AS change_direction), "
                    ":conf, CAST(:ln AS date), CAST(:es AS date), ARRAY['sentinel2'], :scenes, CAST(:ev AS jsonb), "
                    "CAST(:mv AS jsonb), :cv)"
                ),
                [r | {"cv": events.CASCADE_VERSION} for r in rows],
            )
        audit.append(
            conn,
            ACTOR,
            "change.run",
            {
                "aoi": aoi,
                "cascade_version": events.CASCADE_VERSION,
                "inputs_key": key,
                "dates": len(s.dates),
                "events": len(rows),
                "by_type": by_type,
            },
        )
    state_f.write_text(json.dumps({"key": key, "cascade_version": events.CASCADE_VERSION}), encoding="utf-8")
    return {"aoi": aoi, "ard_built": n_ard, "dates": len(s.dates), "events": len(rows), "by_type": by_type}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in sys.argv[1:] or sorted(aois):
        print(json.dumps(run(name)))
