"""Attach Landsat 8/9 evidence to stored change events (M9). `python -m worker.landsat_evidence <aoi>` (make landsat).

An independent optical sensor (30 m, QA_PIXEL mask) checks each Sentinel-2 event: per clear Landsat date, index
medians over the event footprint; the median of the year before last_normal_date vs the year after
earliest_supported_date (whole years, so seasons average out). Those deltas go through the same type rules
(change/classify.py); `agrees` = Landsat gives the same type. Guards against S2-only artefacts (processing-baseline
seams, sensor quirks: PS §2.2.3). Stored in evidence.landsat; `landsat` joins `sensors` when it agrees.
Confidence is NOT touched (DESIGN.md §2.2: M5 calibration decides). One audit entry per AOI run.
"""

import json
import logging
import sys
from datetime import date, timedelta

import numpy as np
from numpy.typing import NDArray
from pyproj import Transformer
from rasterio.features import rasterize
from shapely import wkt
from shapely.ops import transform
from sqlalchemy import text

from change.classify import Evidence, classify
from configs.settings import T, aois, paths
from features import indices
from governance import audit
from index import db
from ingest import scene, tiling
from worker.ingest import ACTOR, timed

log = logging.getLogger("worker.landsat_evidence")
NAMES = ("ndvi", "mndwi", "ndbi", "bsi")


def window(dates: list[str], ln: str | None, es: str) -> tuple[list[int], list[int]]:
    """Indices of dates in (ln - WINDOW_DAYS, ln] and [es, es + WINDOW_DAYS)."""
    w = timedelta(days=T.landsat.WINDOW_DAYS)
    d = [date.fromisoformat(x) for x in dates]
    e = date.fromisoformat(es)
    before = [i for i, x in enumerate(d) if ln and date.fromisoformat(ln) - w < x <= date.fromisoformat(ln)]
    after = [i for i, x in enumerate(d) if e <= x < e + w]
    return before, after


def side_median(vals: NDArray[np.float32], valid: NDArray[np.bool_], idx: list[int]) -> float | None:
    """vals/valid: (T, P) footprint pixels per date. Median over dates of the per-date footprint median, counting
    only dates with >= MIN_VALID_FRAC valid pixels; None with fewer than MIN_OBS such dates."""
    per = [
        float(np.median(vals[i][valid[i]]))
        for i in idx
        if valid[i].size and valid[i].mean() >= T.landsat.MIN_VALID_FRAC
    ]
    return float(np.median(per)) if len(per) >= T.landsat.MIN_OBS else None


def verdict(
    change_type: str, deltas: dict[str, float | None], water: dict, shape: dict
) -> tuple[str | None, bool | None]:
    """(Landsat type, agrees). No claim for "other" or when a side lacks clear dates."""
    if any(v is None for v in deltas.values()):
        return None, None
    ev = Evidence(
        d_ndvi=deltas["ndvi"],  # type: ignore[arg-type]
        d_mndwi=deltas["mndwi"],  # type: ignore[arg-type]
        d_ndbi=deltas["ndbi"],  # type: ignore[arg-type]
        d_bsi=deltas["bsi"],  # type: ignore[arg-type]
        water_pre=water.get("pre", 0.0),
        water_post=water.get("post", 0.0),
        elongation=shape.get("elongation", 1.0),
        width_px=shape.get("width_px", 1.0),
    )
    lt = classify(ev)[0]
    return lt, (None if change_type == "other" else lt == change_type)


def load_stack(aoi: str):
    """All Landsat dates of the AOI as index stacks (float16, NaN = unusable) on the native 30 m grid."""
    folders = sorted(p.parent for p in (paths.raw / "landsat" / aoi).glob("*/scene.json"))
    dates, stack, tr, epsg = [], {n: [] for n in NAMES}, None, None
    for f in folders:
        obs = scene.load_landsat(f)
        ix = indices.compute(obs.refl)
        for n in NAMES:
            stack[n].append(np.where(obs.valid, ix[n], np.nan).astype(np.float16))
        dates.append(obs.date)
        tr, epsg = obs.transform, obs.epsg
    return dates, {n: np.stack(v) for n, v in stack.items()}, tr, epsg


def run(aoi: str) -> dict:
    if not (paths.raw / "landsat" / aoi).exists():
        return {"aoi": aoi, "landsat": "not staged"}
    with timed("landsat_load", aoi=aoi):
        dates, stack, tr, epsg = load_stack(aoi)
    to_utm = Transformer.from_crs(4326, epsg, always_xy=True).transform
    shape_hw = stack["ndvi"].shape[1:]
    assert epsg == tiling.aoi_epsg(aois[aoi].bbox), "Landsat staged on another UTM zone than the AOI grid"
    with db.engine().connect() as conn:
        evs = (
            conn.execute(
                text(
                    "SELECT e.event_id, e.change_type::text AS change_type, e.last_normal_date::text AS ln, "
                    "e.earliest_supported_date::text AS es, ST_AsText(e.geom) AS geom, "
                    "e.evidence->'water_frac' AS water, e.evidence->'shape' AS shape "
                    "FROM change_events e JOIN tiles t USING (tile_id) WHERE t.aoi = :a"
                ),
                {"a": aoi},
            )
            .mappings()
            .all()
        )
    rows, stats = [], {"agree": 0, "disagree": 0, "no_claim": 0}
    with timed("landsat_evidence", aoi=aoi, n=len(evs)):
        for ev in evs:
            g = transform(to_utm, wkt.loads(ev["geom"]))
            fp = rasterize(
                [(g, 1)], out_shape=shape_hw, transform=tr, all_touched=True, dtype="uint8"
            ).astype(bool)
            before, after = window(dates, ev["ln"], ev["es"])
            deltas: dict[str, float | None] = {}
            for n in NAMES:
                v = stack[n][:, fp].astype(np.float32)
                ok = np.isfinite(v)
                b, a = side_median(v, ok, before), side_median(v, ok, after)
                deltas[n] = None if a is None or b is None else round(a - b, 4)
            lt, agrees = verdict(ev["change_type"], deltas, ev["water"] or {}, ev["shape"] or {})
            stats["agree" if agrees is True else "disagree" if agrees is False else "no_claim"] += 1
            out = {
                "n_before": len(before),
                "n_after": len(after),
                "footprint_px": int(fp.sum()),
                "deltas": deltas,
                "type": lt,
                "agrees": agrees,
            }
            rows.append({"i": ev["event_id"], "l": json.dumps(out), "add": agrees is True})
    with db.engine().begin() as conn:
        conn.execute(
            text(
                "UPDATE change_events SET evidence = evidence || jsonb_build_object('landsat', CAST(:l AS jsonb)), "
                "sensors = CASE WHEN :add AND NOT ('landsat' = ANY(sensors)) THEN sensors || ARRAY['landsat'] "
                "WHEN NOT :add THEN array_remove(sensors, 'landsat') ELSE sensors END WHERE event_id = :i"
            ),
            rows,
        )
        audit.append(
            conn,
            ACTOR,
            "change.landsat_evidence",
            {"aoi": aoi, "events": len(rows), "dates": len(dates), **stats, "params": T.landsat.model_dump()},
        )
    return {"aoi": aoi, "events": len(rows), "dates": len(dates), **stats}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in sys.argv[1:] or sorted(aois):
        print(json.dumps(run(name)))
