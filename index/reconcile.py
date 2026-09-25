"""Rebuild Qdrant from PostGIS (DESIGN.md §2.6). `python -m index.reconcile` (make reconcile)."""

import json

import numpy as np
from sqlalchemy import text

from index import db, qdrant

BATCH = 1000


KEYS = ("comp_id", "tile_id", "aoi", "season", "start_date", "end_date", "q_mean", "n_obs")


def payload(row) -> dict:
    """Qdrant payload for one composite row (mapping); shared with the ingest worker so both write the same."""
    return {k: row[k] if isinstance(row[k], int | float) else str(row[k]) for k in KEYS}


SELECT = """
    SELECT c.comp_id, c.tile_id, t.aoi, c.season, c.start_date, c.end_date, c.q_mean, c.n_obs, c.text_vec
    FROM composites c JOIN tiles t USING (tile_id)
    WHERE c.text_vec IS NOT NULL
"""


def rebuild() -> int:
    qdrant.recreate_collection()
    n = 0
    with db.engine().connect() as conn:
        rows = conn.execution_options(stream_results=True, yield_per=BATCH).execute(text(SELECT))
        for part in rows.partitions():
            qdrant.upsert(
                [
                    (
                        f"{r.tile_id}|{r.season}",
                        payload(r._mapping),
                        {"text_vec": np.asarray(r.text_vec, np.float32)},
                    )
                    for r in part
                ]
            )
            n += len(part)
    return n


if __name__ == "__main__":
    print(json.dumps({"reconciled_points": rebuild()}))
