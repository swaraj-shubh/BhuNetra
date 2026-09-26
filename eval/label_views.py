"""One labelling view per sampled event (M5): before | after zoomed on the event outline + index timeline.

Before = latest composite ending before the earliest supported date; after = the same season name one year after
the change (seasonally comparable), else the first later composite. Monsoon (JJAS) composites are patchy, so they
are used only when nothing else has >= MIN_VALID of the window. Nothing is recomputed: stored events + composites.

    python eval/label_views.py         # needs DATABASE_URL; writes data/labels/m5_views/<event_id>.png
"""

import csv
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from pyproj import Transformer
from rasterio.windows import from_bounds
from shapely import wkt
from shapely.ops import transform
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent))
from timeline import H as TL_H
from timeline import plot

from configs.settings import aois, paths
from index import db
from ingest import tiling
from ingest.standardise import rgb_quicklook

SAMPLE = paths.root / "eval" / "labels" / "m5_sample.csv"
OUT = paths.data / "labels" / "m5_views"
PANEL, MIN_HALF_M, MARGIN_M, MIN_VALID = 380, 400.0, 250.0, 0.6


def seasons(aoi: str) -> list[dict]:
    return sorted(
        (json.loads(f.read_text(encoding="utf-8")) for f in (paths.composites / aoi).glob("*.json")),
        key=lambda s: s["start_date"],
    )


def crop(aoi: str, season: str, bounds_m) -> tuple[np.ndarray, float] | None:
    with rasterio.open(paths.composites / aoi / f"{season}.tif") as d:
        w = from_bounds(*bounds_m, transform=d.transform)
        r, g, b = (
            d.read(d.descriptions.index(n) + 1, window=w, boundless=True, fill_value=np.nan)
            for n in ("red", "green", "blue")
        )
    valid = float(np.isfinite(r).mean())
    return (rgb_quicklook(r, g, b), valid) if valid > 0 else None


def pick(aoi: str, cands: list[dict], bounds_m) -> tuple[str, np.ndarray] | None:
    ordered = [s for s in cands if not s["season"].endswith("JJAS")] + [
        s for s in cands if s["season"].endswith("JJAS")
    ]
    best = None
    for s in ordered:
        c = crop(aoi, s["season"], bounds_m)
        if c and c[1] >= MIN_VALID:
            return s["season"], c[0]
        if c and (best is None or c[1] > best[2]):
            best = (s["season"], c[0], c[1])
    return (best[0], best[1]) if best else None


def panel(img: np.ndarray, fp, bounds_m, title: str) -> Image.Image:
    im = Image.fromarray(img).resize((PANEL, PANEL), Image.Resampling.NEAREST)
    g = ImageDraw.Draw(im)
    x0, y0, x1, y1 = bounds_m
    sx, sy = PANEL / (x1 - x0), PANEL / (y1 - y0)
    for poly in getattr(fp, "geoms", [fp]):
        g.line([((x - x0) * sx, (y1 - y) * sy) for x, y in poly.exterior.coords], fill=(255, 230, 0), width=2)
    g.rectangle([0, 0, PANEL, 16], fill=(0, 0, 0))
    g.text((4, 2), title, fill="white")
    return im


def view(ev: dict, blind: bool = False) -> Image.Image:
    aoi = ev["tile_id"].split("_")[0]
    epsg = tiling.aoi_epsg(aois[aoi].bbox)
    fp = transform(Transformer.from_crs(4326, epsg, always_xy=True).transform, wkt.loads(ev["geom"]))
    cx, cy = fp.centroid.x, fp.centroid.y
    fx0, fy0, fx1, fy1 = fp.bounds
    half = max(MIN_HALF_M, max(fx1 - fx0, fy1 - fy0) / 2 + MARGIN_M)
    bounds_m = (cx - half, cy - half, cx + half, cy + half)
    es = date.fromisoformat(ev["earliest_supported_date"])
    ss = seasons(aoi)
    before = pick(aoi, [s for s in reversed(ss) if date.fromisoformat(s["end_date"]) < es], bounds_m)
    name = before[0][4:] if before else None
    same = [s for s in ss if s["season"] == f"{es.year + 1}{name}"] if name else []
    later = [s for s in ss if date.fromisoformat(s["start_date"]) >= es + timedelta(days=180)]
    after = pick(aoi, same + [s for s in later if s not in same], bounds_m)
    tl = plot(ev).crop((0, 0, 900, TL_H))
    if blind:  # held-out: hide the system's type/confidence so the human label is independent
        ImageDraw.Draw(tl).rectangle([0, 0, 900, 28], fill="white")
    im = Image.new("RGB", (900, PANEL + TL_H + 10), "white")
    for j, (lbl, p) in enumerate((("BEFORE", before), ("AFTER", after))):
        if p:
            im.paste(
                panel(p[1], fp, bounds_m, f"{lbl} {p[0]} composite · {2 * half / 1000:.1f} km wide"),
                (40 + j * 440, 0),
            )
        else:
            ImageDraw.Draw(im).text((40 + j * 440 + 100, PANEL // 2), f"{lbl}: no composite", fill="black")
    im.paste(tl, (0, PANEL + 10))
    return im


def main() -> None:
    with open(SAMPLE, encoding="utf-8") as f:
        split_of = {r["event_id"]: r["split"] for r in csv.DictReader(f)}
    ids = list(split_of)
    with db.engine().connect() as conn:
        rows = conn.execute(
            text(
                "SELECT e.event_id, e.tile_id, e.change_type::text AS change_type, e.direction::text AS direction, "
                "e.confidence, e.last_normal_date::text AS last_normal_date, "
                "e.earliest_supported_date::text AS earliest_supported_date, e.cascade_version, e.evidence, "
                "ST_AsText(e.geom) AS geom FROM change_events e WHERE e.event_id = ANY(:ids)"
            ),
            {"ids": ids},
        ).mappings()
        evs = [dict(r) for r in rows]
    OUT.mkdir(parents=True, exist_ok=True)
    for i, ev in enumerate(evs, 1):
        view(ev, blind=split_of[ev["event_id"]] == "heldout").save(OUT / f"{ev['event_id']}.png")
        if i % 50 == 0:
            print(f"{i}/{len(evs)}", flush=True)
    print(json.dumps({"views": len(evs), "dir": str(OUT)}))


if __name__ == "__main__":
    main()
