"""eo_vec for every seasonal composite of an AOI (M7). `python -m worker.embed_eo <aoi>` (make embed-eo AOI=).

Same tiles as text_vec (those with a text_vec; VALID_FRAC_EMB). Idempotent: a season whose rows already carry this
eo_vec model version is skipped. Write order: PostGIS commit (+ audit) -> Qdrant update_vectors (DESIGN.md §2.6).
"""

import json
import logging
import sys

import numpy as np
import rasterio
from sqlalchemy import text

from configs.settings import aois, paths
from features import embed_prithvi
from governance import audit
from index import db, qdrant
from ingest import tiling
from worker.ingest import ACTOR, timed

log = logging.getLogger("worker.embed_eo")


def run(aoi: str, full_sync: bool = True) -> dict:
    """full_sync: re-push every eo_vec of the AOI to Qdrant at the end (bulk runs); incremental jobs skip it."""
    a = aois[aoi]
    epsg = tiling.aoi_epsg(a.bbox)
    bounds = tiling.aoi_bounds_m(a.bbox, epsg)
    tiles = {t.tile_id: t for t in tiling.tiles(aoi, bounds, epsg)}
    lat, lon = (a.bbox[1] + a.bbox[3]) / 2, (a.bbox[0] + a.bbox[2]) / 2
    ver = embed_prithvi.version()
    counts = {"built": 0, "skipped": 0}
    with db.engine().connect() as conn:
        seasons = conn.execute(
            text(
                "SELECT c.season, min(c.start_date)::text, min(c.end_date)::text, "
                "bool_and(c.model_versions->>'eo_vec' IS NOT DISTINCT FROM :v) FROM composites c "
                "JOIN tiles t USING (tile_id) WHERE t.aoi = :a AND c.text_vec IS NOT NULL GROUP BY c.season ORDER BY 1"
            ),
            {"a": aoi, "v": ver},
        ).all()
    for season, start, end, done in seasons:
        if done:
            counts["skipped"] += 1
            continue
        with db.engine().connect() as conn:
            ids = [
                r[0]
                for r in conn.execute(
                    text(
                        "SELECT c.tile_id FROM composites c JOIN tiles t USING (tile_id) "
                        "WHERE t.aoi = :a AND c.season = :s AND c.text_vec IS NOT NULL ORDER BY 1"
                    ),
                    {"a": aoi, "s": season},
                )
            ]
        side = json.loads((paths.composites / aoi / f"{season}.json").read_text(encoding="utf-8"))
        with rasterio.open(paths.composites / aoi / side["file"]) as ds:  # any GSD (10 m S2, ~30 m organiser)
            gsd, names = ds.res[0], list(ds.descriptions)
        tile_len = round(tiling.T.tiling.TILE_M / gsd)
        offs = [
            (round((bounds[3] - tiles[i].bounds_m[3]) / gsd), round((tiles[i].bounds_m[0] - bounds[0]) / gsd))
            for i in ids
        ]
        used = embed_prithvi.bands_used(names)
        mid = str(np.datetime64(start) + (np.datetime64(end) - np.datetime64(start)) // 2)
        with timed("embed_eo", aoi=aoi, season=season, n=len(ids)):
            vecs = embed_prithvi.embed_composite(
                paths.composites / aoi / side["file"], offs, tile_len, mid, lat, lon
            )
        with db.engine().begin() as conn:
            # one statement per season (pg8000 executemany = one round trip per row: ~80 s/season)
            conn.execute(
                text(
                    "UPDATE composites c SET eo_vec = ARRAY(SELECT jsonb_array_elements_text(j.value)::real), "
                    "model_versions = c.model_versions || CAST(:m AS jsonb) "
                    "FROM jsonb_each(CAST(:j AS jsonb)) j WHERE c.tile_id = j.key AND c.season = :s"
                ),
                {
                    "j": json.dumps(
                        {t: [round(float(x), 6) for x in v] for t, v in zip(ids, vecs, strict=True)}
                    ),
                    "m": json.dumps(
                        {"eo_vec": ver}
                        | ({"eo_vec_bands": used} if used != list(embed_prithvi.BANDS) else {})
                    ),
                    "s": season,
                },
            )
            audit.append(
                conn, ACTOR, "embed.eo_vec", {"aoi": aoi, "season": season, "tiles": len(ids), "model": ver}
            )
        qdrant.set_vectors([(f"{t}|{season}", {"eo_vec": v}) for t, v in zip(ids, vecs, strict=True)])
        counts["built"] += 1
        log.info(json.dumps({"aoi": aoi, "season": season, "tiles": len(ids)}))
    counts["qdrant_synced"] = sync(aoi) if full_sync else 0
    return {"aoi": aoi, "model": ver, **counts}


def sync(aoi: str) -> int:
    """Re-push eo_vec from PostGIS (source of truth) to Qdrant, so a failed upsert after a commit heals on rerun."""
    n = 0
    with db.engine().connect() as conn:
        rows = conn.execution_options(stream_results=True).execute(
            text(
                "SELECT c.tile_id, c.season, c.eo_vec FROM composites c JOIN tiles t USING (tile_id) "
                "WHERE t.aoi = :a AND c.eo_vec IS NOT NULL"
            ),
            {"a": aoi},
        )
        for part in rows.partitions(2000):
            qdrant.set_vectors(
                [(f"{r.tile_id}|{r.season}", {"eo_vec": np.asarray(r.eo_vec, np.float32)}) for r in part]
            )
            n += len(part)
    return n


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # bulk load: pause HNSW indexing (every update_vectors re-indexed 349k points), rebuild once at the end
    qdrant.set_indexing(False)
    try:
        for name in sys.argv[1:] or sorted(aois):
            print(json.dumps(run(name)))
    finally:
        qdrant.set_indexing(True)
