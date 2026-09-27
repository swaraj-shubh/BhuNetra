"""Convert our label CSVs into the organiser-style inputs of eval/run.py (needs DATABASE_URL for geometries).

  python eval/to_eval_format.py cases   <events.csv> <out.geojson>   # event_id,real,true_type (m5 labels)
  python eval/to_eval_format.py queries <tiles.csv>  <out.json>      # tile_id,labels (m1 own tiles)

cases: one case per labelled event (real yes -> change, no -> no_change, unsure skipped), geometry = event
footprint, window = last_normal_date - PAD .. earliest_supported_date + PAD, earliest_date = the event's own
date (so the date error only checks plumbing). Event-derived cases measure event precision; recall on them is
optimistic (every case sits on an event). Organiser cases do not have that bias.
queries: one query per class with >= MIN_POS tiles; relevant = the labelled tiles. Unlabelled tiles in a ranking
count as misses, so P@k is a lower bound (pooled judgements needed for a real number).
"""

import csv
import json
import sys
from datetime import date, timedelta
from pathlib import Path

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.own_tiles import CLASSES, MIN_POS
from index import db

PAD = timedelta(days=180)


def cases(src: Path, out: Path) -> int:
    lab = {r["event_id"]: r for r in csv.DictReader(src.open(encoding="utf-8")) if r["real"] in ("yes", "no")}
    with db.engine().connect() as conn:
        rows = conn.execute(
            text(
                "SELECT event_id, change_type::text AS change_type, last_normal_date, earliest_supported_date, "
                "ST_AsGeoJSON(geom) AS geom FROM change_events WHERE event_id = ANY(:ids)"
            ),
            {"ids": list(lab)},
        ).mappings()
        feats = []
        for r in rows:
            lb = lab[r["event_id"]]
            es: date = r["earliest_supported_date"]
            feats.append(
                {
                    "type": "Feature",
                    "geometry": json.loads(r["geom"]),
                    "properties": {
                        "case_id": r["event_id"],
                        "label": "change" if lb["real"] == "yes" else "no_change",
                        "t1": str((r["last_normal_date"] or es) - PAD),
                        "t2": str(es + PAD),
                        "type": (lb.get("true_type") or r["change_type"]) if lb["real"] == "yes" else None,
                        "earliest_date": str(es) if lb["real"] == "yes" else None,
                    },
                }
            )
    out.write_text(json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")
    return len(feats)


def queries(src: Path, out: Path) -> int:
    by_cls: dict[str, set[str]] = {}
    for r in csv.DictReader(src.open(encoding="utf-8")):
        for c in filter(None, r["labels"].split(";")):
            by_cls.setdefault(c, set()).add(r["tile_id"])
    with db.engine().connect() as conn:
        geom = dict(
            conn.execute(
                text("SELECT tile_id, ST_AsGeoJSON(geom) FROM tiles WHERE tile_id = ANY(:ids)"),
                {"ids": sorted(set().union(*by_cls.values()))},
            ).all()
        )
    qs = [
        {"id": c, "query": CLASSES[c], "relevant": [json.loads(geom[t]) for t in sorted(ts)]}
        for c, ts in sorted(by_cls.items())
        if len(ts) >= MIN_POS
    ]
    out.write_text(json.dumps({"queries": qs}), encoding="utf-8")
    return len(qs)


if __name__ == "__main__":
    kind, src, dst = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    print(f"{kind}: {({'cases': cases, 'queries': queries}[kind])(src, dst)} written to {dst}")
