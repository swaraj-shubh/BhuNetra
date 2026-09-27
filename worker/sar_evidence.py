"""Attach Sentinel-1 log-ratio evidence to stored change events (M9). `python -m worker.sar_evidence <aoi>`.

Per event: VV/VH median (linear) over the footprint for the N_BEFORE S1 acquisitions up to last_normal_date and the
N_AFTER from earliest_supported_date; log-ratio in dB; agreement with the optical type (change/sar.py). Stored in
evidence.s1; `sentinel1` joins `sensors` when S1 agrees. Confidence is NOT touched (DESIGN.md §2.2: M5 calibration
decides whether S1 agreement enters it). One audit entry per AOI run.
"""

import json
import logging
import sys
from bisect import bisect_left, bisect_right
from functools import lru_cache

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.windows import from_bounds
from shapely import wkt
from shapely.ops import transform
from sqlalchemy import text

from change import sar
from configs.settings import T, aois, paths
from governance import audit
from index import db
from ingest import tiling
from worker.ingest import ACTOR, timed

log = logging.getLogger("worker.sar_evidence")


def s1_dates(aoi: str) -> list[tuple[str, dict]]:
    out = []
    for f in sorted((paths.raw / "sentinel1" / aoi).glob("*/scene.json")):
        meta = json.loads(f.read_text(encoding="utf-8"))
        out.append((meta["date"], meta))
    return out


@lru_cache(maxsize=256)
def _open(path: str):
    return rasterio.open(path)


def _read(aoi: str, meta: dict, band: str, geom_utm) -> tuple[np.ndarray, np.ndarray] | None:
    ds = _open(str(paths.raw / "sentinel1" / aoi / meta["date"] / meta["bands"][band]["file"]))
    win = from_bounds(*geom_utm.bounds, transform=ds.transform).round_offsets().round_lengths()
    if win.width < 1 or win.height < 1:
        return None
    arr = ds.read(1, window=win, boundless=True, fill_value=meta["bands"][band]["nodata"])
    mask = rasterize(
        [(geom_utm, 1)], out_shape=arr.shape, transform=ds.window_transform(win), dtype="uint8"
    ).astype(bool)
    return arr, mask


def evidence(aoi: str, dates: list[tuple[str, dict]], ev: dict, geom_utm) -> dict:
    names = [d for d, _ in dates]
    ln, es = ev["last_normal_date"], ev["earliest_supported_date"]
    before = dates[max(0, bisect_right(names, ln) - T.sar.N_BEFORE) : bisect_right(names, ln)] if ln else []
    after = dates[bisect_left(names, es) : bisect_left(names, es) + T.sar.N_AFTER]
    out: dict = {"orbit": None, "before": [d for d, _ in before], "after": [d for d, _ in after]}
    if before or after:
        p = (before or after)[0][1]["properties"]
        out["orbit"] = f"{p.get('sat:relative_orbit')}{str(p.get('sat:orbit_state', ''))[:1]}"
    for band in ("VV", "VH"):
        med = []
        for group in (before, after):
            stack, mask = [], None
            for _, meta in group:
                r = _read(aoi, meta, band, geom_utm)
                if r is not None:
                    stack.append(r[0])
                    mask = r[1]
            nodata = group[0][1]["bands"][band]["nodata"] if group else -32768
            med.append(sar.stack_median_db(stack, mask, nodata) if stack and mask is not None else None)
        out[f"{band.lower()}_before_db"], out[f"{band.lower()}_after_db"] = med
        dlt = sar.log_ratio_db(*med)
        out[f"d{band.lower()}_db"] = None if dlt is None else round(dlt, 2)
    out["agrees"] = sar.agrees(ev["change_type"], out["dvv_db"], out["dvh_db"])
    return out


def run(aoi: str) -> dict:
    dates = s1_dates(aoi)
    if not dates:
        return {"aoi": aoi, "s1": "no staged Sentinel-1"}
    epsg = tiling.aoi_epsg(aois[aoi].bbox)
    to_utm = Transformer.from_crs(4326, epsg, always_xy=True).transform
    with db.engine().connect() as conn:
        evs = (
            conn.execute(
                text(
                    "SELECT e.event_id, e.change_type::text AS change_type, e.last_normal_date::text AS last_normal_date, "
                    "e.earliest_supported_date::text AS earliest_supported_date, ST_AsText(e.geom) AS geom "
                    "FROM change_events e JOIN tiles t USING (tile_id) WHERE t.aoi = :a"
                ),
                {"a": aoi},
            )
            .mappings()
            .all()
        )
    rows, stats = [], {"agree": 0, "disagree": 0, "no_claim": 0}
    with timed("sar_evidence", aoi=aoi, n=len(evs)):
        for ev in evs:
            e = evidence(aoi, dates, dict(ev), transform(to_utm, wkt.loads(ev["geom"])))
            stats["agree" if e["agrees"] is True else "disagree" if e["agrees"] is False else "no_claim"] += 1
            rows.append({"i": ev["event_id"], "s1": json.dumps(e), "add": e["agrees"] is True})
    with db.engine().begin() as conn:
        conn.execute(
            text(
                "UPDATE change_events SET evidence = evidence || jsonb_build_object('s1', CAST(:s1 AS jsonb)), "
                "sensors = CASE WHEN :add AND NOT ('sentinel1' = ANY(sensors)) THEN sensors || ARRAY['sentinel1'] "
                "WHEN NOT :add THEN array_remove(sensors, 'sentinel1') ELSE sensors END WHERE event_id = :i"
            ),
            rows,
        )
        audit.append(
            conn,
            ACTOR,
            "change.sar_evidence",
            {"aoi": aoi, "events": len(rows), **stats, "params": T.sar.model_dump()},
        )
    return {"aoi": aoi, "events": len(rows), **stats}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in sys.argv[1:] or sorted(aois):
        print(json.dumps(run(name)))
