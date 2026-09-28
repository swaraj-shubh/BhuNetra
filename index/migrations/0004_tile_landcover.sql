-- Per-tile, per-year land-cover shares from the independent IO annual LULC maps (features/landcover.py).
-- Idempotent: safe to re-run (make db-migrate). Used at query time as a lookup only (DESIGN.md §2.1).
CREATE TABLE IF NOT EXISTS tile_landcover (
    tile_id     text NOT NULL REFERENCES tiles (tile_id),
    year        integer NOT NULL,
    fracs       jsonb NOT NULL,                     -- group -> share of the tile's classified pixels
    lineage     jsonb NOT NULL,                     -- input file + sha256, class groups
    PRIMARY KEY (tile_id, year)
);
