"""DB-backed job queue (DESIGN.md §4 "Jobs"; M9 incremental ingest). Heavy work happens here, never in the API (§2.1).

    python -m worker.jobs                      # worker loop (docker compose service `worker`)
    python -m worker.jobs run --path P --aoi A # one incremental ingest, synchronously (make ingest-scene)

Job kind `ingest_scene` {path, aoi}: `path` is a staged acquisition folder (data/raw/<source>/<aoi>/<date>) or
GeoTIFF/COG file(s) (staged first as source `organiser`, M3b). Steps, each timed into jobs.timings:
  scene (validate + q) -> season (recomposite + re-embed only that season: searchable) -> change (AOI cascade).
Nothing else is rebuilt: other seasons, other AOIs and the rest of the index are untouched.
"""

import argparse
import json
import logging
import time
from pathlib import Path

from sqlalchemy import text

from configs.settings import aois, paths
from features import composites
from index import db
from ingest import local, scene
from worker import change, ingest

log = logging.getLogger("worker.jobs")
POLL_S = 3


def resolve(path: str, aoi: str) -> Path:
    """Staged folder for `path`; stages raw files first. Only paths inside data/ are accepted."""
    p = (paths.root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if paths.data.resolve() not in p.parents and p != paths.data.resolve():
        raise ValueError(f"path must be inside {paths.data}")
    if (p / "scene.json").exists():
        return p
    files = [p] if p.is_file() else sorted(p.glob("*.tif"))
    if not files:
        raise ValueError(f"no scene.json or .tif at {p}")
    res = local.stage(files, aoi=aoi, source="organiser")
    return paths.raw / "organiser" / aoi / res[0]["date"]


def ingest_one(path: str, aoi: str, progress=lambda t: None) -> dict:
    if aoi not in aois:
        raise ValueError(f"unknown AOI {aoi}")
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    folder = resolve(path, aoi)
    tiles, _ = ingest._tiles(aoi)
    status = ingest.ingest_scene(folder, tiles)
    timings["scene_s"] = round(time.perf_counter() - t0, 1)
    progress(timings)
    season = composites.season_of(scene.read_meta(folder)["date"])
    with db.engine().connect() as conn:
        accepted = [
            paths.root / r[0]
            for r in conn.execute(
                text("SELECT raw_path FROM scenes WHERE aoi = :a AND status = 'accepted'"), {"a": aoi}
            )
        ]
    same_season = sorted(f for f in accepted if composites.season_of(scene.read_meta(f)["date"]) == season)
    season_status = ingest.ingest_season(aoi, season, same_season, tiles)
    timings["searchable_s"] = round(time.perf_counter() - t0, 1)  # PS metric: incremental ingest time
    progress(timings)
    ch = change.run(aoi)
    timings["change_s"] = round(time.perf_counter() - t0, 1)
    return {
        "folder": str(folder.relative_to(paths.root)),
        "scene": status,
        "season": season,
        "season_status": season_status,
        "change": ch,
        "timings": timings,
    }


def _claim(conn):
    return conn.execute(
        text(
            "UPDATE jobs SET status = 'running' WHERE job_id = (SELECT job_id FROM jobs WHERE status = 'queued' "
            "ORDER BY created_ts FOR UPDATE SKIP LOCKED LIMIT 1) RETURNING job_id, kind, params"
        )
    ).first()


def _update(job_id, finish: bool = False, **cols) -> None:
    sets = [f"{k} = CAST(:{k} AS jsonb)" if k == "timings" else f"{k} = :{k}" for k in cols]
    sets += ["finished_ts = now()"] if finish else []
    with db.engine().begin() as conn:
        conn.execute(text(f"UPDATE jobs SET {', '.join(sets)} WHERE job_id = :id"), {"id": job_id, **cols})


def loop() -> None:
    log.info("worker ready")
    while True:
        with db.engine().begin() as conn:
            job = _claim(conn)
        if job is None:
            time.sleep(POLL_S)
            continue
        job_id, kind, params = job
        log.info(json.dumps({"job": str(job_id), "kind": kind, "params": params}))
        try:
            if kind != "ingest_scene":
                raise ValueError(f"unknown job kind {kind}")
            res = ingest_one(
                params["path"], params["aoi"], lambda t, j=job_id: _update(j, timings=json.dumps(t))
            )
            _update(job_id, status="done", timings=json.dumps(res["timings"] | {"result": res}), finish=True)
        except Exception as e:
            log.exception("job failed")
            _update(job_id, status="failed", error=f"{type(e).__name__}: {e}"[:2000], finish=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run")
    r.add_argument("--path", required=True)
    r.add_argument("--aoi", required=True)
    a = ap.parse_args()
    if a.cmd == "run":
        print(json.dumps(ingest_one(a.path, a.aoi), default=str))
    else:
        loop()
