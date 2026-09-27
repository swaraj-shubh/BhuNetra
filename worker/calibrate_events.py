"""Apply the M5 calibration to an AOI's stored events (a logged, versioned step: DESIGN.md §2.2).

    python -m worker.calibrate_events <aoi> [...]      (make calibrate AOI=...)

Runs after the S1 and Landsat evidence (its features). confidence := calibrated P(real); the rule score stays in
evidence.raw_confidence (taken from there on re-runs, so repeated application is idempotent); evidence.calibrated
= model version. No configs/calibration.json -> events stay uncalibrated. One audit entry per AOI run.
"""

import json
import logging
import sys

import numpy as np
from sqlalchemy import text

from change.calibrate import event_features, load_model
from configs.settings import aois, paths
from governance import audit
from index import db
from worker.ingest import ACTOR, timed

log = logging.getLogger("worker.calibrate_events")
MODEL = paths.root / "configs" / "calibration.json"


def run(aoi: str) -> dict:
    model = load_model(MODEL)
    if model is None:
        return {"aoi": aoi, "calibration": "none (configs/calibration.json missing)"}
    with db.engine().connect() as conn:
        evs = (
            conn.execute(
                text(
                    "SELECT e.event_id, e.change_type::text AS change_type, e.confidence, e.evidence "
                    "FROM change_events e JOIN tiles t USING (tile_id) WHERE t.aoi = :a AND e.analyst_status = 'unreviewed'"
                ),
                {"a": aoi},
            )
            .mappings()
            .all()
        )
    if not evs:
        return {"aoi": aoi, "events": 0}
    with timed("calibrate", aoi=aoi, n=len(evs)):
        raw = [float(e["evidence"].get("raw_confidence", e["confidence"])) for e in evs]
        X = np.array(
            [event_features(r, e["change_type"], e["evidence"]) for r, e in zip(raw, evs, strict=True)]
        )
        p = model.predict(X)
    rows = [
        {
            "i": e["event_id"],
            "c": round(float(pi), 4),
            "e": json.dumps({"raw_confidence": r, "calibrated": model.version}),
        }
        for e, r, pi in zip(evs, raw, p, strict=True)
    ]
    with db.engine().begin() as conn:
        conn.execute(
            text(
                "UPDATE change_events SET confidence = :c, evidence = evidence || CAST(:e AS jsonb) WHERE event_id = :i"
            ),
            rows,
        )
        shown = int((p >= model.operating_conf).sum())
        audit.append(
            conn,
            ACTOR,
            "change.calibrate",
            {
                "aoi": aoi,
                "events": len(rows),
                "version": model.version,
                "operating_conf": model.operating_conf,
                "above_operating_conf": shown,
            },
        )
    return {"aoi": aoi, "events": len(rows), "version": model.version, "above_operating_conf": shown}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in sys.argv[1:] or sorted(aois):
        print(json.dumps(run(name)))
