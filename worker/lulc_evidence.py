"""Independent land-cover evidence for stored change events. `python -m worker.lulc_evidence <aoi>` (make lulc).

Impact Observatory annual LULC v02 (10 m, one map per year, staged read-only in data/raw/io_lulc/<aoi>/<year>/) is
a separately trained global product: neither our cascade nor our labels shaped it. Per event: the class mix inside
the footprint in the full year before the change and the full year after it; the transition implies a type
(built up -> construction, water up/down -> water, bare up -> clearance), "stable" when no class moves by
MIN_FRAC. Stored in evidence.lulc; the calibration model may use it (M5). Roads are below what annual LULC
resolves, so road events get no verdict. Confidence is NOT touched here (DESIGN.md §2.2). One audit entry per run.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import rasterio
from numpy.typing import NDArray
from pyproj import Transformer
from rasterio.features import rasterize
from shapely import wkt
from shapely.ops import transform
from sqlalchemy import text

from configs.settings import T, aois, paths
from governance import audit
from index import db
from worker.ingest import ACTOR, timed

log = logging.getLogger("worker.lulc_evidence")
# io-lulc-annual-v02 classes -> groups (0 nodata, 9 snow, 10 clouds are ignored)
GROUPS = {"water": (1,), "built": (7,), "bare": (8,), "vegetation": (2, 4, 5, 11)}


def class_fracs(lulc: NDArray[np.uint8], fp: NDArray[np.bool_]) -> dict[str, float] | None:
    """Share of each group among the footprint's classified pixels; None when nothing is classified."""
    v = lulc[fp]
    known = np.isin(v, [c for cs in GROUPS.values() for c in cs])
    if not known.any():
        return None
    return {g: round(float(np.isin(v[known], cs).mean()), 4) for g, cs in GROUPS.items()}


def verdict(change_type: str, before: dict | None, after: dict | None) -> tuple[str | None, bool | None]:
    """(LULC-implied type or "stable", agrees). No claim for roads, "other" or missing years."""
    if before is None or after is None:
        return None, None
    d = {g: after[g] - before[g] for g in GROUPS}
    m = T.lulc.MIN_FRAC
    if d["built"] >= m:
        lt = "construction"
    elif abs(d["water"]) >= m:
        lt = "water"
    elif d["bare"] >= m:
        lt = "clearance"
    elif max(abs(x) for x in d.values()) < m:
        lt = "stable"
    else:
        lt = "other"
    if change_type in ("road", "other"):
        return lt, None
    return lt, lt == change_type


def years(aoi: str) -> dict[int, Path]:
    return {int(p.parent.name): p for p in (paths.raw / "io_lulc" / aoi).glob("*/LULC.tif")}


def run(aoi: str) -> dict:
    ys = years(aoi)
    if not ys:
        return {"aoi": aoi, "lulc": "not staged"}
    maps, tr, crs = {}, None, None
    with timed("lulc_load", aoi=aoi):
        for y, f in ys.items():
            with rasterio.open(f) as ds:
                maps[y], tr, crs = ds.read(1), ds.transform, ds.crs
    to_map = Transformer.from_crs(4326, crs, always_xy=True).transform
    shape_hw = next(iter(maps.values())).shape
    with db.engine().connect() as conn:
        evs = (
            conn.execute(
                text(
                    "SELECT e.event_id, e.change_type::text AS change_type, e.earliest_supported_date AS es, "
                    "ST_AsText(e.geom) AS geom FROM change_events e JOIN tiles t USING (tile_id) WHERE t.aoi = :a"
                ),
                {"a": aoi},
            )
            .mappings()
            .all()
        )
    rows, stats = [], {"agree": 0, "disagree": 0, "stable": 0, "no_claim": 0}
    with timed("lulc_evidence", aoi=aoi, n=len(evs)):
        for ev in evs:
            fp = rasterize(
                [(transform(to_map, wkt.loads(ev["geom"])), 1)],
                out_shape=shape_hw,
                transform=tr,
                all_touched=True,
                dtype="uint8",
            ).astype(bool)
            yb, ya = ev["es"].year - 1, ev["es"].year + 1  # full years either side of the change
            before = class_fracs(maps[yb], fp) if yb in maps else None
            after = class_fracs(maps[ya], fp) if ya in maps else None
            lt, agrees = verdict(ev["change_type"], before, after)
            stats[
                "stable"
                if lt == "stable"
                else "agree"
                if agrees
                else "disagree"
                if agrees is False
                else "no_claim"
            ] += 1
            out = {
                "before_year": yb,
                "after_year": ya,
                "before": before,
                "after": after,
                "type": lt,
                "agrees": agrees,
            }
            rows.append({"i": ev["event_id"], "l": json.dumps(out)})
    with db.engine().begin() as conn:
        conn.execute(
            text(
                "UPDATE change_events SET evidence = evidence || jsonb_build_object('lulc', CAST(:l AS jsonb)) "
                "WHERE event_id = :i"
            ),
            rows,
        )
        audit.append(
            conn,
            ACTOR,
            "change.lulc_evidence",
            {"aoi": aoi, "events": len(rows), "years": sorted(ys), **stats, "params": T.lulc.model_dump()},
        )
    return {"aoi": aoi, "events": len(rows), **stats}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in sys.argv[1:] or sorted(aois):
        print(json.dumps(run(name)))
