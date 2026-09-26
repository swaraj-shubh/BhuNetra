"""Automatic (weak) change labels for M5 from independent annual products (team decision 2026-09-26).

Per stored event: Impact Observatory LULC composition inside the event footprint, the full year before
`earliest_supported_date` vs the year after it (both within the staged 2018-2023), plus new OSM road length for road events.

    outcome  tp           the footprint changed the way the event's type says
             fp_nochange  composition stable (<= STABLE_MAX) and no new road: a false alarm
             fp_wrongtype real change, but not of the stated type
             ambiguous    in between; excluded from calibration/metrics
             n/a          years outside IO coverage or too few valid labels

**Finding 2026-09-26 (calibration blocks only): IO LULC is too temporally smoothed to label our events** -
66-79 % of footprints get an identical class for every pixel in both years (even 200-2000 px events), and obvious
mine expansions come out 'no change'. Kept as a diagnostic, NOT used to fit calibration. Also: IO LULC is also Sentinel-2-derived, so it is not fully independent of our cascade, and its
crops/rangeland/bare classes are noisy. Final held-out numbers come from the human-verified sample, not from this file.

    python eval/weak_labels.py          # needs DATABASE_URL; writes data/labels/m5_weak.csv
"""

import csv
import json
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.windows import from_bounds
from shapely import wkt
from shapely.geometry import LineString
from shapely.ops import transform
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent))
import split

from configs.settings import T, aois, eval_cfg, paths
from governance import lineage
from index import db
from ingest import tiling

IO = eval_cfg["io_lulc"]
WL = eval_cfg["weak_labels"]
C = IO["classes"]
OUT = paths.data / "labels" / "m5_weak.csv"


@lru_cache(maxsize=32)
def lulc(aoi: str, year: int) -> tuple[np.ndarray, rasterio.Affine]:
    with rasterio.open(paths.raw / "io_lulc" / aoi / str(year) / "LULC.tif") as ds:
        return ds.read(1), ds.transform


@lru_cache(maxsize=16)
def roads(aoi: str, snapshot: str) -> list[LineString]:
    f = paths.raw / "osm" / aoi / f"roads_{snapshot}.geojson"
    if not f.exists():
        return []
    epsg = tiling.aoi_epsg(aois[aoi].bbox)
    to_utm = Transformer.from_crs(4326, epsg, always_xy=True).transform
    keep = set(T.context.ROAD_CLASSES)
    fc = json.loads(f.read_text(encoding="utf-8"))
    return [
        transform(to_utm, LineString(x["geometry"]["coordinates"]))
        for x in fc["features"]
        if x["properties"].get("highway") in keep and x["geometry"]["type"] == "LineString"
    ]


def fractions(arr: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    vals = arr[mask]
    return {k: float((vals == v).mean()) if vals.size else 0.0 for k, v in C.items()}


def road_len(lines: list[LineString], fp) -> float:
    return float(sum(line.intersection(fp).length for line in lines if line.intersects(fp)))


def label(change_type: str, delta: dict[str, float], total: float, new_road_m: float) -> str:
    m = WL["MIN_FRAC"]
    real = total >= m or new_road_m >= WL["ROAD_MIN_M"]
    if total <= WL["STABLE_MAX"] and new_road_m < WL["ROAD_MIN_M"]:
        return "fp_nochange"
    rules = {
        "construction": delta["built"] >= m,
        "clearance": -delta["trees"] >= m,
        "water": abs(delta["water"] + delta["flooded_vegetation"]) >= m,
        "road": new_road_m >= WL["ROAD_MIN_M"],
        "other": total >= m,
    }
    if rules[change_type]:
        return "tp"
    return "fp_wrongtype" if real else "ambiguous"


def run() -> list[dict]:
    frozen = json.loads(split.OUT.read_text(encoding="utf-8"))
    y0, y1 = IO["years"]
    with db.engine().connect() as conn:
        events = (
            conn.execute(
                text(
                    "SELECT e.event_id, e.tile_id, t.aoi, e.change_type::text AS change_type, e.confidence, "
                    "e.last_normal_date, e.earliest_supported_date, ST_AsText(e.geom) AS geom, "
                    "e.evidence->'confidence_components' AS comp FROM change_events e JOIN tiles t USING (tile_id) "
                    "ORDER BY e.event_id"
                )
            )
            .mappings()
            .all()
        )
    rows = []
    for e in events:
        aoi = e["aoi"]
        epsg = tiling.aoi_epsg(aois[aoi].bbox)
        # full calendar year before the change (the IO map of the change year already contains post-change months)
        before = e["earliest_supported_date"].year - 1 if e["last_normal_date"] else None
        after = e["earliest_supported_date"].year + 1
        base = {
            "event_id": e["event_id"],
            "aoi": aoi,
            "tile_id": e["tile_id"],
            "split": split.split_of(e["tile_id"], frozen),
            "change_type": e["change_type"],
            "raw_confidence": round(float(e["confidence"]), 4),
        }
        if before is None or before < y0 or after > y1:
            rows.append(base | {"outcome": "n/a", "reason": f"years {before}->{after} outside IO {y0}-{y1}"})
            continue
        fp = transform(Transformer.from_crs(4326, epsg, always_xy=True).transform, wkt.loads(e["geom"]))
        a0, tf = lulc(aoi, before)
        a1, _ = lulc(aoi, after)
        win = from_bounds(*fp.bounds, transform=tf).round_offsets().round_lengths()
        r0, c0 = int(win.row_off), int(win.col_off)
        h, w = max(1, int(win.height)), max(1, int(win.width))
        wtf = rasterio.windows.transform(win, tf)
        inside = rasterize([(fp, 1)], out_shape=(h, w), transform=wtf, dtype="uint8").astype(bool)
        b0, b1 = a0[r0 : r0 + h, c0 : c0 + w], a1[r0 : r0 + h, c0 : c0 + w]
        valid = inside & ~np.isin(b0, IO["nodata"]) & ~np.isin(b1, IO["nodata"])
        vfrac = float(valid.sum() / max(1, inside.sum()))
        if vfrac < WL["MIN_VALID_FRAC"]:
            rows.append(base | {"outcome": "n/a", "reason": f"valid IO fraction {vfrac:.2f}"})
            continue
        f0, f1 = fractions(b0, valid), fractions(b1, valid)
        delta = {k: f1[k] - f0[k] for k in C}
        total = 0.5 * sum(abs(v) for v in delta.values())
        rb = road_len(roads(aoi, f"{before}-01-01"), fp)
        ra = road_len(roads(aoi, f"{min(after + 1, 2026)}-01-01"), fp)
        new_road = max(0.0, ra - rb)
        rows.append(
            base
            | {
                "outcome": label(e["change_type"], delta, total, new_road),
                "reason": f"IO {before}->{after}",
                "total_change": round(total, 3),
                "d_built": round(delta["built"], 3),
                "d_trees": round(delta["trees"], 3),
                "d_water": round(delta["water"] + delta["flooded_vegetation"], 3),
                "new_road_m": round(new_road, 1),
                "area_px": int(inside.sum()),
            }
        )
    return rows


if __name__ == "__main__":
    rows = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cols = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)
    print(json.dumps({"n": len(rows), "written": str(OUT), "generated": lineage.now_ts()}))
    for key in ("outcome",):
        print(Counter((r["split"], r["change_type"], r[key]) for r in rows).most_common())
