-- M6: stored searches, so /feedback can re-rank a query by id (Rocchio). Idempotent.
CREATE TABLE IF NOT EXISTS queries (
    query_id    uuid PRIMARY KEY,
    text        text NOT NULL,
    request     jsonb NOT NULL,      -- aoi, date_range, k as sent
    parsed      jsonb NOT NULL,
    created_ts  timestamptz NOT NULL DEFAULT now()
);
