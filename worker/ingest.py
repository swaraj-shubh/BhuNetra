"""Ingest one AOI's staged Sentinel-2 into PostGIS + Qdrant (M2). `python -m worker.ingest <aoi>` (make ingest).

Resumable and idempotent: a scene whose scene_id is already in `scenes` is skipped; a season whose composite rows
already carry the same observation fingerprints is skipped. Write order per unit: PostGIS commit (with its audit
entry, same transaction) -> Qdrant upsert (DESIGN.md §2.6, §2.7, §10).
"""

import json
import logging
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import rasterio
from shapely.geometry import Polygon
from sqlalchemy import Connection, text

from configs.settings import PROCESSING_VERSION, T, aois, paths
from features import composites, context, embed_clip
from governance import audit, lineage
from index import db, qdrant, reconcile
from ingest import quality, scene, tiling
from ingest.standardise import rgb_quicklook

log = logging.getLogger("worker.ingest")
ACTOR = "worker"


@contextmanager
def timed(stage: str, **ctx):
    """Stage timings -> manifests/build.json (DESIGN.md §10; numbers go into the report)."""
    t0 = time.perf_counter()
    yield
    dt = round(time.perf_counter() - t0, 2)
    f = paths.manifests / "build.json"
    data = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    data.setdefault("timings", {}).setdefault(stage, []).append({**ctx, "s": dt, "ts": lineage.now_ts()})
    f.write_text(json.dumps(data, indent=1), encoding="utf-8")
    log.info(json.dumps({"stage": stage, "s": dt, **ctx}))


def _tiles(aoi: str) -> tuple[list[tiling.Tile], int]:
    a = aois[aoi]
    epsg = tiling.aoi_epsg(a.bbox)
    return list(tiling.tiles(aoi, tiling.aoi_bounds_m(a.bbox, epsg), epsg)), epsg


def register_tiles(conn: Connection, aoi: str, tiles: list[tiling.Tile], epsg: int) -> None:
    conn.execute(
        text(
            "INSERT INTO tiles (tile_id, aoi, epsg, row_idx, col_idx, bounds_m, geom) "
            "VALUES (:id, :aoi, :epsg, :r, :c, :b, ST_GeomFromText(:wkt, 4326)) ON CONFLICT DO NOTHING"
        ),
        [
            {
                "id": t.tile_id,
                "aoi": aoi,
                "epsg": epsg,
                "r": t.row,
                "c": t.col,
                "b": list(t.bounds_m),
                "wkt": Polygon(tiling.footprint_lonlat(t.bounds_m, epsg)).wkt,
            }
            for t in tiles
        ],
    )


def register_context(aoi: str, tiles: list[tiling.Tile], epsg: int) -> str:
    """Per-tile distance to water/road (M4). Skipped when stored lineage has the same inputs and params."""
    ctx = context.compute(aoi, tiles, epsg)
    with db.engine().begin() as conn:
        stored = {
            r[0]: r[1]
            for r in conn.execute(
                text(
                    "SELECT DISTINCT ON (c.target) c.target, c.lineage FROM tile_context c "
                    "JOIN tiles t USING (tile_id) WHERE t.aoi = :a"
                ),
                {"a": aoi},
            )
        }
        if all(stored.get(k) == lin for k, (_, lin) in ctx.items()):
            return "skipped"
        conn.execute(
            text("DELETE FROM tile_context WHERE tile_id IN (SELECT tile_id FROM tiles WHERE aoi = :a)"),
            {"a": aoi},
        )
        for target, (dist, lin) in ctx.items():
            conn.execute(
                text(
                    "INSERT INTO tile_context (tile_id, target, dist_m, lineage) "
                    "VALUES (:t, :g, :d, CAST(:l AS jsonb))"
                ),
                [
                    {"t": t.tile_id, "g": target, "d": float(d), "l": json.dumps(lin)}
                    for t, d in zip(tiles, dist, strict=True)
                ],
            )
        audit.append(
            conn,
            ACTOR,
            "ingest.context",
            {"aoi": aoi, "targets": {k: v[1]["n_features"] for k, v in ctx.items()}},
        )
    return "written"


def register_landcover(aoi: str, tiles: list[tiling.Tile]) -> str:
    """Per-tile land-cover shares per year (features/landcover.py). Skipped when stored lineage is unchanged."""
    from features import landcover

    lc = landcover.compute(aoi, tiles)
    if not lc:
        return "not staged"
    with db.engine().begin() as conn:
        stored = dict(
            conn.execute(
                text(
                    "SELECT DISTINCT ON (l.year) l.year, l.lineage FROM tile_landcover l JOIN tiles t USING (tile_id) "
                    "WHERE t.aoi = :a"
                ),
                {"a": aoi},
            ).all()
        )
        if all(stored.get(y) == lin for y, (_, lin) in lc.items()):
            return "skipped"
        conn.execute(
            text("DELETE FROM tile_landcover WHERE tile_id IN (SELECT tile_id FROM tiles WHERE aoi = :a)"),
            {"a": aoi},
        )
        for year, (fracs, lin) in lc.items():
            conn.execute(
                text(
                    "INSERT INTO tile_landcover (tile_id, year, fracs, lineage) "
                    "VALUES (:t, :y, CAST(:f AS jsonb), CAST(:l AS jsonb))"
                ),
                [
                    {"t": t.tile_id, "y": year, "f": json.dumps(f), "l": json.dumps(lin)}
                    for t, f in zip(tiles, fracs, strict=True)
                    if f is not None
                ],
            )
        audit.append(conn, ACTOR, "ingest.landcover", {"aoi": aoi, "years": sorted(lc)})
    return "written"


def ingest_scene(folder: Path, tiles: list[tiling.Tile]) -> str:
    meta = scene.read_meta(folder)
    scene_id = "+".join(meta["source_items"])
    with db.engine().begin() as conn:
        if conn.execute(text("SELECT 1 FROM scenes WHERE scene_id = :s"), {"s": scene_id}).first():
            return "skipped"
    fp = scene.fingerprint(folder)
    base = {
        "s": scene_id,
        "src": meta["sensor"],
        "aoi": meta["aoi"],
        "d": meta["date"],
        "ts": meta.get("datetime"),
        "epsg": meta["epsg"],
        "wkt": Polygon(tiling.footprint_lonlat(tuple(meta["bounds"]), meta["epsg"])).wkt,
        "raw": str(folder.relative_to(paths.root)),
        "sha": json.dumps({b["file"]: b["sha256"] for b in meta["bands"].values()} | {"scene.json": fp}),
        "props": json.dumps(meta.get("properties", {})),
    }
    insert = text(
        "INSERT INTO scenes (scene_id, source, aoi, acq_date, acq_ts, epsg, footprint, raw_path, sha256, "
        "properties, status, reject_reason) VALUES (:s, :src, :aoi, :d, :ts, :epsg, ST_GeomFromText(:wkt, 4326), "
        ":raw, CAST(:sha AS jsonb), CAST(:props AS jsonb), :status, :reason)"
    )
    try:
        obs = scene.load(folder)
    except scene.SceneRejected as e:  # a bad scene is stored as rejected, never crashes the batch (§10)
        with db.engine().begin() as conn:
            conn.execute(insert, base | {"status": "rejected", "reason": str(e)})
            audit.append(conn, ACTOR, "ingest.scene.reject", {"scene_id": scene_id, "reason": str(e)})
        return "rejected"

    vf, cn = quality.tile_fractions(obs.valid, obs.cloud, obs.transform, tiles)
    sun = obs.sun_elev if obs.sun_elev is not None else np.nan
    q = quality.q_score(vf, np.nan, cn, sun)
    lin = json.dumps({"scene_fingerprint": fp, "processing_version": PROCESSING_VERSION})
    with db.engine().begin() as conn:
        conn.execute(insert, base | {"status": "accepted", "reason": None})
        conn.execute(
            text(
                "INSERT INTO observations (obs_id, scene_id, tile_id, valid_frac, shift_px, cloud_near, sun_elev, "
                "q, lineage) SELECT u.id, :s, u.t, u.vf, NULL, u.cn, :sun, u.q, CAST(:lin AS jsonb) FROM unnest("
                "CAST(:ids AS uuid[]), CAST(:tids AS text[]), CAST(:vfs AS real[]), CAST(:cns AS real[]), "
                "CAST(:qs AS real[])) AS u(id, t, vf, cn, q)"
            ),
            {  # one statement per scene: executemany is a round trip per row with pg8000
                "s": scene_id,
                "sun": obs.sun_elev,
                "lin": lin,
                "ids": [str(db.uuid7()) for _ in tiles],
                "tids": [t.tile_id for t in tiles],
                "vfs": vf.tolist(),
                "cns": cn.tolist(),
                "qs": q.tolist(),
            },
        )
        audit.append(
            conn,
            ACTOR,
            "ingest.scene",
            {
                "scene_id": scene_id,
                "fingerprint": fp,
                "tiles": len(tiles),
                "q_ge_qmin": int((q >= T.quality.Q_MIN).sum()),
            },
        )
    return "accepted"


def _chips(tif: Path, tiles: list[tiling.Tile]) -> list[np.ndarray]:
    with rasterio.open(tif) as d:
        r, g, b = (d.read(d.descriptions.index(n) + 1) for n in ("red", "green", "blue"))
        out = []
        for t in tiles:
            w = tiling.window(t, d.transform)
            sl = (
                slice(round(w.row_off), round(w.row_off + w.height)),
                slice(round(w.col_off), round(w.col_off + w.width)),
            )
            # VERIFY (M1 part 2): same 8-bit stretch for embedding as for display; EuroSAT JPGs used their own.
            out.append(rgb_quicklook(r[sl], g[sl], b[sl]))
    return out


def ingest_season(aoi: str, season: str, folders: list[Path], tiles: list[tiling.Tile]) -> str:
    fps = sorted(scene.fingerprint(f) for f in folders)
    with db.engine().connect() as conn:
        have = conn.execute(
            text(
                "SELECT lineage->'observations' AS obs, model_versions FROM composites c JOIN tiles t "
                "USING (tile_id) WHERE t.aoi = :aoi AND c.season = :s LIMIT 1"
            ),
            {"aoi": aoi, "s": season},
        ).first()
    # same inputs AND same embedding model -> nothing to do; a new text_vec model re-embeds (ADR 0002 fallback)
    if (
        have is not None
        and sorted(o["fingerprint"] for o in have.obs) == fps
        and have.model_versions.get("text_vec") == embed_clip.version()
    ):
        return "skipped"

    with timed("composite", aoi=aoi, season=season, n_obs=len(folders)):
        side = composites.build(aoi, season, folders, tiles)
    st = side["tiles"]
    emb_idx = [
        i for i, v in enumerate(st["valid_frac"]) if v >= T.quality.VALID_FRAC_EMB and st["n_obs"][i] > 0
    ]
    with timed("embed_clip", aoi=aoi, season=season, n_tiles=len(emb_idx)):
        vecs = embed_clip.embed_images(
            _chips(paths.composites / aoi / side["file"], [tiles[i] for i in emb_idx])
        )
    vec_of = dict(zip(emb_idx, vecs, strict=True))
    mv = json.dumps({"text_vec": embed_clip.version()})
    lin = lineage.record(
        "composite",
        inputs={
            "observations": side["lineage"]["observations"],
            "file": f"data/composites/{aoi}/{side['file']}",
        },
        params={k: v for k, v in side["lineage"].items() if k != "observations"},
        models={"text_vec": embed_clip.version()},
    ) | {"observations": side["lineage"]["observations"]}

    rows = [
        {
            "id": db.uuid7(),
            "t": t.tile_id,
            "s": season,
            "a": side["start_date"],
            "b": side["end_date"],
            "n": st["n_obs"][i],
            "q": st["q_mean"][i],
            "v": vec_of[i].tolist() if i in vec_of else None,
            "mv": mv,
            "lin": json.dumps(lin | {"valid_frac": st["valid_frac"][i]}),
        }
        for i, t in enumerate(tiles)
    ]
    with timed("postgis_commit", aoi=aoi, season=season), db.engine().begin() as conn:
        conn.execute(
            text(
                "DELETE FROM composites WHERE season = :s AND tile_id IN (SELECT tile_id FROM tiles WHERE aoi = :aoi)"
            ),
            {"s": season, "aoi": aoi},
        )
        conn.execute(
            text(
                "INSERT INTO composites (comp_id, tile_id, season, start_date, end_date, n_obs, q_mean, text_vec, "
                "model_versions, lineage) SELECT u.id, u.t, :s, CAST(:a AS date), CAST(:b AS date), u.n, u.q, "
                "CAST(u.v AS real[]), CAST(:mv AS jsonb), CAST(u.lin AS jsonb) FROM unnest(CAST(:ids AS uuid[]), "
                "CAST(:tids AS text[]), CAST(:ns AS int[]), CAST(:qs AS real[]), CAST(:vs AS text[]), "
                "CAST(:lins AS text[])) AS u(id, t, n, q, v, lin)"
            ),
            {  # one statement per season (see ingest_scene); vectors travel as '{...}' array literals
                "s": season,
                "a": side["start_date"],
                "b": side["end_date"],
                "mv": mv,
                "ids": [str(r["id"]) for r in rows],
                "tids": [r["t"] for r in rows],
                "ns": [r["n"] for r in rows],
                "qs": [r["q"] for r in rows],
                "vs": [None if r["v"] is None else "{" + ",".join(map(repr, r["v"])) + "}" for r in rows],
                "lins": [r["lin"] for r in rows],
            },
        )
        audit.append(
            conn,
            ACTOR,
            "ingest.composite",
            {
                "aoi": aoi,
                "season": season,
                "observations": fps,
                "embedded_tiles": len(emb_idx),
                "models": json.loads(mv),
            },
        )
    qdrant.ensure_collection()  # after the PostGIS commit (§2.6)
    with timed("qdrant_upsert", aoi=aoi, season=season):
        for k in range(0, len(emb_idx), reconcile.BATCH):
            part = []
            for i in emb_idx[k : k + reconcile.BATCH]:
                r = rows[i]
                row = dict(
                    zip(
                        reconcile.KEYS,
                        (r["id"], r["t"], aoi, season, r["a"], r["b"], r["q"], r["n"]),
                        strict=True,
                    )
                )
                part.append((f"{r['t']}|{season}", reconcile.payload(row), {"text_vec": vec_of[i]}))
            qdrant.upsert(part)
    return "built"


def ingest_aoi(aoi: str) -> dict:
    tiles, epsg = _tiles(aoi)
    with db.engine().begin() as conn:
        register_tiles(conn, aoi, tiles, epsg)
    with timed("context", aoi=aoi):
        counts_ctx = register_context(aoi, tiles, epsg)
    with timed("landcover", aoi=aoi):
        register_landcover(aoi, tiles)
    src_dir = paths.raw / aois[aoi].optical_source / aoi  # one optical source per AOI time series (M3b)
    folders = sorted(p for p in src_dir.iterdir() if (p / "scene.json").exists())
    counts: dict[str, int] = defaultdict(int)
    with timed("scenes", aoi=aoi, n=len(folders)):
        for f in folders:
            counts[f"scene_{ingest_scene(f, tiles)}"] += 1
    with db.engine().connect() as conn:
        accepted = {
            r[0]
            for r in conn.execute(
                text("SELECT raw_path FROM scenes WHERE aoi = :a AND status = 'accepted'"), {"a": aoi}
            )
        }
    by_season: dict[str, list[Path]] = defaultdict(list)
    for f in folders:
        if str(f.relative_to(paths.root)) in accepted:
            by_season[composites.season_of(f.name)].append(f)
    for season in sorted(by_season):
        counts[f"season_{ingest_season(aoi, season, by_season[season], tiles)}"] += 1
        log.info(json.dumps({"aoi": aoi, "season": season, **counts}))
    # a failed Qdrant upsert after a committed season is skipped on rerun; surface it (fix: make reconcile)
    with db.engine().connect() as conn:
        in_db = conn.execute(
            text(
                "SELECT count(*) FROM composites c JOIN tiles t USING (tile_id) "
                "WHERE t.aoi = :a AND c.text_vec IS NOT NULL"
            ),
            {"a": aoi},
        ).scalar()
    in_qdrant = qdrant.count({"aoi": aoi})
    if in_db != in_qdrant:
        log.warning(
            json.dumps(
                {"aoi": aoi, "postgis_vectors": in_db, "qdrant_points": in_qdrant, "fix": "make reconcile"}
            )
        )
    return dict(counts) | {"context": counts_ctx, "postgis_vectors": in_db, "qdrant_points": in_qdrant}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in sys.argv[1:] or sorted(aois):
        print(json.dumps({"aoi": name, **ingest_aoi(name)}))
