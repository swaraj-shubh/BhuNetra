-- M4: per-tile spatial context (features/context.py). Idempotent: safe to re-run (make db-migrate).
CREATE TABLE IF NOT EXISTS tile_context (
    tile_id     text NOT NULL REFERENCES tiles (tile_id),
    target      text NOT NULL,                      -- water | road (configs/query.yaml targets)
    dist_m      real NOT NULL,                      -- metres from tile footprint to nearest feature; 'Infinity' = none
    lineage     jsonb NOT NULL,                     -- inputs (file + sha256), params, feature count
    PRIMARY KEY (tile_id, target)
);
CREATE INDEX IF NOT EXISTS tile_context_target_dist ON tile_context (target, dist_m);
