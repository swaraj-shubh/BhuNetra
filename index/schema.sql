-- SIH26227 PostGIS schema v1 (DESIGN.md §2.6: PostGIS is the source of truth; Qdrant is rebuildable).
-- Applied by the db container on first start (docker/compose.yaml) and by `make db-migrate`.

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TYPE change_type AS ENUM ('construction', 'clearance', 'water', 'road', 'other');
CREATE TYPE change_direction AS ENUM ('appearance', 'disappearance', 'expansion', 'contraction');
CREATE TYPE analyst_status AS ENUM ('unreviewed', 'confirmed', 'rejected');

-- One row per staged acquisition (data/raw/<source>/<aoi>/<key>/scene.json)
CREATE TABLE scenes (
    scene_id        text PRIMARY KEY,                 -- STAC item id(s) joined with '+', or product name
    source          text NOT NULL,                    -- sentinel2 | sentinel1 | landsat | bhuvan | ...
    aoi             text NOT NULL,
    acq_date        date,
    acq_ts          timestamptz,
    epsg            integer NOT NULL,
    footprint       geometry(Polygon, 4326),
    raw_path        text NOT NULL,
    sha256          jsonb NOT NULL,                   -- {band_file: sha256}
    properties      jsonb NOT NULL DEFAULT '{}',      -- cloud, orbit, sun elevation, processing baseline
    status          text NOT NULL DEFAULT 'accepted' CHECK (status IN ('accepted', 'rejected')),
    reject_reason   text,
    ingested_ts     timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX scenes_aoi_date ON scenes (aoi, source, acq_date);
CREATE INDEX scenes_footprint ON scenes USING gist (footprint);

-- Fixed AOI grid (ingest/tiling.py): same tile_id = same place across dates and sensors
CREATE TABLE tiles (
    tile_id         text PRIMARY KEY,                 -- <aoi>_<utmzone>_<row:04d>_<col:04d>
    aoi             text NOT NULL,
    epsg            integer NOT NULL,
    row_idx         integer NOT NULL,
    col_idx         integer NOT NULL,
    bounds_m        double precision[4] NOT NULL,     -- xmin, ymin, xmax, ymax in epsg
    geom            geometry(Polygon, 4326) NOT NULL,
    UNIQUE (aoi, row_idx, col_idx)
);
CREATE INDEX tiles_geom ON tiles USING gist (geom);

-- Tile x acquisition quality (DESIGN.md §7 q formula)
CREATE TABLE observations (
    obs_id          uuid PRIMARY KEY,                 -- UUIDv7
    scene_id        text NOT NULL REFERENCES scenes (scene_id),
    tile_id         text NOT NULL REFERENCES tiles (tile_id),
    valid_frac      real NOT NULL,
    shift_px        real,
    cloud_near      real,
    sun_elev        real,
    q               real NOT NULL,
    lineage         jsonb NOT NULL,
    UNIQUE (scene_id, tile_id)
);
CREATE INDEX observations_tile_q ON observations (tile_id, q);

-- Seasonal composites + their vectors (stored here so index/reconcile.py can rebuild Qdrant)
CREATE TABLE composites (
    comp_id         uuid PRIMARY KEY,                 -- UUIDv7
    tile_id         text NOT NULL REFERENCES tiles (tile_id),
    season          text NOT NULL,                    -- e.g. 2024-DJF
    start_date      date NOT NULL,
    end_date        date NOT NULL,
    n_obs           integer NOT NULL,
    q_mean          real NOT NULL,
    text_vec        real[],
    eo_vec          real[],
    model_versions  jsonb NOT NULL DEFAULT '{}',
    lineage         jsonb NOT NULL,
    UNIQUE (tile_id, season)
);

-- Change events: fixed type + fixed calibrated confidence (DESIGN.md §2.2)
CREATE SEQUENCE event_seq;
CREATE TABLE change_events (
    event_id                text PRIMARY KEY DEFAULT ('CHG-' || lpad(nextval('event_seq')::text, 6, '0')),
    tile_id                 text NOT NULL REFERENCES tiles (tile_id),
    geom                    geometry(MultiPolygon, 4326) NOT NULL,
    change_type             change_type NOT NULL,
    direction               change_direction NOT NULL,
    confidence              real NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    last_normal_date        date,
    earliest_supported_date date NOT NULL,
    sensors                 text[] NOT NULL,
    source_scene_ids        text[] NOT NULL,
    evidence                jsonb NOT NULL,           -- index deltas, persistence run, S1 log-ratio, validator
    model_versions          jsonb NOT NULL,
    cascade_version         text NOT NULL,
    analyst_status          analyst_status NOT NULL DEFAULT 'unreviewed',
    created_ts              timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX change_events_geom ON change_events USING gist (geom);
CREATE INDEX change_events_query ON change_events (change_type, earliest_supported_date, confidence);

CREATE TABLE feedback (
    feedback_id     bigserial PRIMARY KEY,
    target_type     text NOT NULL CHECK (target_type IN ('site', 'event')),
    target_id       text NOT NULL,
    decision        text NOT NULL CHECK (decision IN ('confirm', 'reject')),
    reason          text,
    query_id        text,
    actor           text NOT NULL,
    model_versions  jsonb NOT NULL DEFAULT '{}',
    ts              timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE jobs (
    job_id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    kind            text NOT NULL,
    params          jsonb NOT NULL DEFAULT '{}',
    status          text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'done', 'failed')),
    timings         jsonb NOT NULL DEFAULT '{}',
    error           text,
    created_ts      timestamptz NOT NULL DEFAULT now(),
    finished_ts     timestamptz
);

-- Hash-chained audit log (DESIGN.md §7): entry_hash = sha256(prev_hash || ts || actor || action || canonical_json)
-- Computed by governance/audit.py in the same transaction as the state change. Rows are append-only.
CREATE TABLE audit_log (
    seq             bigserial PRIMARY KEY,
    ts              timestamptz NOT NULL,
    actor           text NOT NULL,
    action          text NOT NULL,
    payload         jsonb NOT NULL,
    prev_hash       char(64) NOT NULL,
    entry_hash      char(64) NOT NULL UNIQUE
);

CREATE FUNCTION audit_log_append_only() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'audit_log is append-only (% blocked)', TG_OP;
END;
$$;
CREATE TRIGGER audit_log_no_update BEFORE UPDATE OR DELETE ON audit_log
    FOR EACH ROW EXECUTE FUNCTION audit_log_append_only();
CREATE TRIGGER audit_log_no_truncate BEFORE TRUNCATE ON audit_log
    FOR EACH STATEMENT EXECUTE FUNCTION audit_log_append_only();

-- M4 migrations (index/migrations/*.sql) are applied on top of this file by `make db-migrate`.
