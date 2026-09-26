"""Timeline plot for change events (M3 exit: "timeline with breakpoint for a known site").

    python -m eval.timeline CHG-000123 [CHG-...]
    python -m eval.timeline --aoi hasdeo --type clearance --top 3      # highest-confidence events of a type
    python -m eval.timeline --aoi dholera --lonlat 72.19,22.25         # event(s) covering a point

Reads stored events from PostGIS (nothing is recomputed, §2.2) and writes eval/reports/timelines/<event_id>.png:
per-date medians of each index over the event footprint, last normal date and earliest supported date marked,
plus before/after chips from the seasonal composites.
"""

import argparse
import io
from datetime import date

from PIL import Image, ImageDraw
from sqlalchemy import text

from api.main import chip
from configs.settings import paths
from features.composites import season_of
from index import db

OUT = paths.root / "eval" / "reports" / "timelines"
COLOURS = {"ndvi": (34, 139, 34), "ndbi": (200, 60, 60), "mndwi": (30, 90, 200), "bsi": (160, 110, 40)}
W, H, PAD = 900, 360, 50


def _x(d: str, d0: date, span: int) -> float:
    return PAD + (date.fromisoformat(d) - d0).days / span * (W - 2 * PAD)


def _y(v: float) -> float:
    return H - PAD - (v + 1) / 2 * (H - 2 * PAD)  # indices in [-1, 1]


def plot(ev: dict) -> Image.Image:
    tl = ev["evidence"]["timeline"]
    dates = tl["dates"]
    d0 = date.fromisoformat(dates[0])
    span = max((date.fromisoformat(dates[-1]) - d0).days, 1)
    im = Image.new("RGB", (W, H + 200), "white")
    g = ImageDraw.Draw(im)
    g.rectangle([PAD, PAD, W - PAD, H - PAD], outline=(180, 180, 180))
    for v in (-1, -0.5, 0, 0.5, 1):
        g.line([PAD, _y(v), W - PAD, _y(v)], fill=(235, 235, 235))
        g.text((8, _y(v) - 6), f"{v:+.1f}", fill="black")
    for y in range(d0.year, date.fromisoformat(dates[-1]).year + 1):
        x = _x(f"{y}-01-01", d0, span)
        if x >= PAD:
            g.text((x - 12, H - PAD + 6), str(y), fill="black")
    for k, lbl, col in (
        ("last_normal_date", "last normal", (120, 120, 120)),
        ("earliest_supported_date", "earliest supported", (220, 0, 0)),
    ):
        if ev.get(k):
            x = _x(ev[k], d0, span)
            g.line([x, PAD, x, H - PAD], fill=col, width=2)
            g.text((x + 3, PAD + (0 if col[0] == 220 else 14)), f"{lbl} {ev[k]}", fill=col)
    for i, (name, col) in enumerate(COLOURS.items()):
        pts = [
            (_x(d, d0, span), _y(v)) for d, v in zip(dates, tl.get(name, []), strict=False) if v is not None
        ]
        if len(pts) > 1:
            g.line(pts, fill=col, width=2)
        for p in pts:
            g.ellipse([p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2], fill=col)
        g.text((W - PAD - 70, PAD + 4 + 14 * i), name.upper(), fill=col)
    conf = ev["confidence"]
    g.text(
        (PAD, 12),
        f"{ev['event_id']}  {ev['change_type']} / {ev['direction']}  conf {conf:.2f} (uncalibrated)  "
        f"tile {ev['tile_id']}  cascade {ev['cascade_version']}",
        fill="black",
    )
    # composites strictly before and after the break season (last normal and break can share a season)
    aoi = ev["tile_id"].split("_")[0]
    seasons = sorted(f.stem for f in (paths.composites / aoi).glob("*.json"))
    k = (
        seasons.index(season_of(ev["earliest_supported_date"]))
        if season_of(ev["earliest_supported_date"]) in seasons
        else -1
    )
    picks = (
        ("before", seasons[k - 1] if k > 0 else None),
        ("after", seasons[k + 1] if 0 <= k < len(seasons) - 1 else None),
    )
    for j, (lbl, season) in enumerate(picks):
        if season:
            try:
                png = chip(ev["tile_id"], season, size=180).body
                im.paste(Image.open(io.BytesIO(png)), (PAD + j * 220, H + 10))
                g.text((PAD + j * 220, H + 192), f"{lbl}: {season} composite", fill="black")
            except Exception:  # noqa: BLE001 - a missing composite must not stop the plot
                g.text((PAD + j * 220, H + 90), f"{lbl}: no composite", fill="black")
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--aoi")
    ap.add_argument("--type")
    ap.add_argument("--lonlat")
    ap.add_argument("--top", type=int, default=3)
    a = ap.parse_args()
    where, params = ["TRUE"], {}
    if a.ids:
        where.append("e.event_id = ANY(:ids)")
        params["ids"] = a.ids
    if a.aoi:
        where.append("t.aoi = :aoi")
        params["aoi"] = a.aoi
    if a.type:
        where.append("e.change_type = CAST(:ty AS change_type)")
        params["ty"] = a.type
    if a.lonlat:
        lon, lat = map(float, a.lonlat.split(","))
        where.append("ST_Intersects(e.geom, ST_SetSRID(ST_Point(:lon, :lat), 4326))")
        params |= {"lon": lon, "lat": lat}
    with db.engine().connect() as conn:
        rows = conn.execute(
            text(
                "SELECT e.event_id, e.tile_id, e.change_type::text AS change_type, e.direction::text AS direction, "
                "e.confidence, e.last_normal_date::text AS last_normal_date, "
                "e.earliest_supported_date::text AS earliest_supported_date, e.cascade_version, e.evidence "
                f"FROM change_events e JOIN tiles t USING (tile_id) WHERE {' AND '.join(where)} "
                "ORDER BY e.confidence DESC LIMIT :n"
            ),
            params | {"n": len(a.ids) or a.top},
        ).mappings()
        evs = [dict(r) for r in rows]
    OUT.mkdir(parents=True, exist_ok=True)
    for ev in evs:
        plot(ev).save(OUT / f"{ev['event_id']}.png")
        print(OUT / f"{ev['event_id']}.png")


if __name__ == "__main__":
    main()
