"""Frozen, stratified event sample for M5 labelling (team decision 2026-09-26: model pre-labels, team corrects,
held-out labelled blind by humans). Uniform within each (split, change_type); commit the output BEFORE labelling.

    python eval/sample_events.py      # needs DATABASE_URL; writes eval/labels/m5_sample.csv
"""

import csv
import json
import random
import sys
from pathlib import Path

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent))
import split

from configs.settings import eval_cfg, paths
from index import db

CFG = eval_cfg["labelling"]
OUT = paths.root / "eval" / "labels" / "m5_sample.csv"


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists: the sample is frozen")
    frozen = json.loads(split.OUT.read_text(encoding="utf-8"))
    with db.engine().connect() as c:
        evs = c.execute(
            text("SELECT event_id, tile_id, change_type::text AS t FROM change_events ORDER BY event_id")
        ).all()
    pop: dict[tuple[str, str], list[str]] = {}
    for eid, tid, t in evs:
        pop.setdefault((split.split_of(tid, frozen), t), []).append(eid)
    rng = random.Random(CFG["SEED"])
    rows = []
    for sp, quota in CFG["quota"].items():
        for t, n in quota.items():
            members = pop.get((sp, t), [])
            for eid in sorted(rng.sample(members, min(n, len(members)))):
                rows.append(
                    {"event_id": eid, "split": sp, "change_type": t, "stratum_population": len(members)}
                )
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(
        json.dumps(
            {"n": len(rows), "per_split": {sp: sum(r["split"] == sp for r in rows) for sp in CFG["quota"]}}
        )
    )


if __name__ == "__main__":
    main()
