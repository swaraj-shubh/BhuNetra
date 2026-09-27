"""Our labels -> organiser-style inputs of eval/run.py. Labels are anchored to frozen geometry, never to live ids.

  python eval/to_eval_format.py snapshot [recovered.json]           # once: freeze the M5 sample (needs DATABASE_URL)
  python eval/to_eval_format.py cases <labels.csv|.jsonl> <out.geojson> [calibration|heldout]
  python eval/to_eval_format.py queries <tiles.csv> <out.json>       # m1 own tiles (needs DATABASE_URL)

snapshot: event ids are not stable across cascade re-runs (change.run replaces unreviewed events), so the sampled
events' footprint + dates are frozen in eval/labels/m5_cases.geojson. `recovered.json` supplies events that were
already replaced (dholera, 2026-09-27: recreated bit-for-bit from the earlier inputs, checked against m5_weak.csv).
cases: one case per labelled sample event (yes -> change, no -> no_change, unsure skipped); window =
last_normal_date - PAD .. earliest_supported_date + PAD. Event-sampled cases measure precision; their recall is
optimistic (each sits where an older cascade fired).
queries: one query per class with >= MIN_POS tiles; relevant = the labelled tiles. Unlabelled tiles in a ranking
count as misses, so P@k is a lower bound.
"""

import csv
import json
import sys
from datetime import date, timedelta
from pathlib import Path

from shapely import wkt
from shapely.geometry import mapping
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.own_tiles import CLASSES, MIN_POS
from index import db

LABELS = Path(__file__).resolve().parent / "labels"
SAMPLE = LABELS / "m5_sample.csv"
SNAPSHOT = LABELS / "m5_cases.geojson"
PAD = timedelta(days=180)


def snapshot(recovered: Path | None) -> int:
    if SNAPSHOT.exists():
        raise SystemExit(f"{SNAPSHOT} exists: the snapshot is frozen")
    sample = list(csv.DictReader(SAMPLE.open(encoding="utf-8")))
    with db.engine().connect() as conn:
        live = {
            r["event_id"]: dict(r)
            for r in conn.execute(
                text(
                    "SELECT event_id, tile_id, change_type::text AS change_type, confidence, cascade_version, "
                    "last_normal_date::text AS last_normal_date, earliest_supported_date::text AS earliest_supported_date, "
                    "ST_AsText(geom) AS geom_wkt FROM change_events WHERE event_id = ANY(:ids)"
                ),
                {"ids": [r["event_id"] for r in sample]},
            ).mappings()
        }
    rec = json.loads(recovered.read_text(encoding="utf-8")) if recovered else {}
    feats, missing = [], []
    for s in sample:
        e = live.get(s["event_id"]) or rec.get(s["event_id"])
        if e is None or e["change_type"] != s["change_type"]:
            missing.append(s["event_id"])
            continue
        feats.append(
            {
                "type": "Feature",
                "geometry": mapping(wkt.loads(e["geom_wkt"])),
                "properties": {
                    "event_id": s["event_id"],
                    "split": s["split"],
                    "system_type": e["change_type"],
                    "confidence": e["confidence"],
                    "tile_id": e["tile_id"],
                    "last_normal_date": e["last_normal_date"],
                    "earliest_supported_date": e["earliest_supported_date"],
                    "source": "live" if s["event_id"] in live else "recovered",
                },
            }
        )
    if missing:
        raise SystemExit(f"{len(missing)} sample events not found or type mismatch, e.g. {missing[:5]}")
    SNAPSHOT.write_text(json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")
    return len(feats)


def read_labels(src: Path) -> dict[str, dict]:
    """csv (m5_prelabels) or jsonl (m5_human, append-only: the last record per event wins)."""
    if src.suffix == ".jsonl":
        out: dict[str, dict] = {}
        for line in src.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["event_id"]] = r
        return out
    return {r["event_id"]: r for r in csv.DictReader(src.open(encoding="utf-8"))}


def cases(src: Path, out: Path, split: str | None = None) -> int:
    lab = read_labels(src)
    feats = []
    for f in json.loads(SNAPSHOT.read_text(encoding="utf-8"))["features"]:
        p = f["properties"]
        lb = lab.get(p["event_id"])
        if not lb or lb["real"] not in ("yes", "no") or (split and p["split"] != split):
            continue
        es = date.fromisoformat(p["earliest_supported_date"])
        ln = date.fromisoformat(p["last_normal_date"]) if p["last_normal_date"] else es
        real = lb["real"] == "yes"
        feats.append(
            {
                "type": "Feature",
                "geometry": f["geometry"],
                "properties": {
                    "case_id": p["event_id"],
                    "label": "change" if real else "no_change",
                    "t1": str(ln - PAD),
                    "t2": str(es + PAD),
                    "type": (lb.get("true_type") or p["system_type"]) if real else None,
                    "earliest_date": str(es) if real else None,
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
    kind, args = sys.argv[1], sys.argv[2:]
    if kind == "snapshot":
        print(f"snapshot: {snapshot(Path(args[0]) if args else None)} events -> {SNAPSHOT}")
    elif kind == "cases":
        print(
            f"cases: {cases(Path(args[0]), Path(args[1]), args[2] if len(args) > 2 else None)} -> {args[1]}"
        )
    else:
        print(f"queries: {queries(Path(args[0]), Path(args[1]))} -> {args[1]}")
