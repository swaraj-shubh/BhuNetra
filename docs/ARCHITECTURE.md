# Architecture note: SIH26227 Satellite Semantic Retrieval & Change Analysis

An offline, on-premises system that makes a satellite-imagery archive searchable **by meaning** (text or example
site) and **by change over time**, with an analyst review loop, a tamper-evident audit trail and signed exports.
Everything runs with the network disabled once models and data are staged.

## 1. Components

```
                ┌──────────── Docker Compose (one host, loopback ports only) ────────────┐
 browser ──────►│ api (FastAPI + React/MapLibre bundle, no CDN)  ──►  db (PostgreSQL 16 + PostGIS) │
                │        │  query path only: encode query, vector/SQL lookup, fuse, dedup          │
                │        └──────────────────────────────────────►  qdrant (vector index, rebuildable)│
                │ worker (same image, GPU): job queue in PostGIS, all heavy compute              │
                └────────────────────────────────────────────────────────────────────────────────┘
 data/raw (read-only AOI extracts + scene.json) ─► worker ─► data/ard, data/composites, data/change
```

| Store | Holds | Rule |
|---|---|---|
| PostGIS | scenes, tiles, observations (tile × date, quality `q`), composites (+ both vectors), change events, feedback, jobs, audit log | **source of truth**; every state change writes a hash-chained audit entry in the same transaction |
| Qdrant | one point per tile × season with named vectors `text_vec` and `eo_vec` + payload | rebuildable from PostGIS (`make reconcile`); write order PostGIS → Qdrant |
| Files | raw extracts (sha256 per band), COG composites and ARD with `lineage` JSON | raw is never written after staging |

## 2. Ingest (all heavy compute, `worker/`)

1. **Validate + standardise** each acquisition: sha256, STAC item, canonical band names, AOI UTM grid (10 m
   anchor), provider scale/offset (Sentinel-2 processing-baseline offset handled per item, ADR 0003).
2. **Masks + quality**: SCL + Sen2Cor cloud probability (dilated) → per tile × date
   `q = valid_frac · clip(1 − shift_px) · (1 − 0.5·cloud_near) · (1 − 0.3·[sun_elev < 25°])`;
   observations with `q < Q_MIN` never enter any time series.
3. **Tiles** are defined in metres (640 m, 320 m stride, grid anchored at a stride multiple so tile ids are stable
   across dates and overlapping AOIs); pixel size = metres / GSD, so any GSD works.
4. **Seasonal composites** (IMD seasons DJF / MAM / JJAS / ON, q-gated median) → two embeddings:
   - `text_vec`: remote-sensing CLIP (GeoRSCLIP ViT-B-32, chosen by the M1 benchmark, ADR 0002), used **only**
     for text → image search;
   - `eo_vec`: Prithvi-EO-2.0-300M-TL (6 bands, 224 px windows, patch tokens pooled per tile), used for
     image → image search, clustering and change evidence. The two spaces are never compared.
5. **Spatial context** per tile (distance to JRC surface water and OSM roads) for "near a river / road" queries.
6. **Change cascade** per tile on the valid time series of NDVI / MNDWI / NDBI / BSI:
   harmonic seasonal model on a 12-month baseline → robust residuals → **persistent break** (K consecutive
   same-sign exceedances; a single-date difference is never a change) → evidence → rule classifier
   (construction / clearance / water / road / other + direction) → confidence → stored event with
   `earliest_supported_date`, `last_normal_date`, source scene ids and model / cascade versions.
7. **Sentinel-1 evidence** (same relative orbit only): VV/VH log-ratio before vs after the break, stored as
   evidence and as an extra sensor when it agrees with the optical type. SAR never enters the optical models.

Confidence is fixed at ingest; it changes only through a logged, versioned re-run of the cascade or calibration.

## 3. Query path (`query/`, `api/`)

```
text query ─► parser {semantic, change intent/type, time, place, sensor}
            ├─ semantic branch: CLIP text embedding → Qdrant text_vec (best season per tile)
            ├─ change branch:  stored events (SQL) of the requested type/time above the operating confidence
            └─ spatial branch: precomputed tile context (near water / road)
            ─► fusion score = w_sim·sim + w_conf·event_conf + w_q·q + w_sp·spatial_fit ─► unique-site dedup ─► sites
site click ─► eo_vec ─► look-alike sites (Qdrant eo_vec) │ UMAP + HDBSCAN clusters (discovery)
```

Every site and event in a response carries `provenance {scene_ids, model_versions, cascade_version}`.

## 4. Review, governance, export

- Analyst queue with before/after chips, index timeline and evidence; confirm / reject (+ reason) → Rocchio
  re-rank live; rejected sites drop out; decisions go to the calibration pool.
- `audit_log`: `entry_hash = sha256(prev_hash ‖ ts ‖ actor ‖ action ‖ payload)`; `GET /audit/verify` recomputes it.
- Exports: zip with GeoJSON + provenance + sha256 manifest, **Ed25519-signed**; `POST /export/verify`.
- Model weights are sha256-checked against `manifests/models.yaml` before loading; API keys with
  viewer / analyst / admin roles.

## 5. Incremental ingestion

`POST /ingest {path}` queues a job; the worker validates the new acquisition, recomputes only the affected season
(composite + both embeddings; other seasons and AOIs untouched), which makes it searchable, then refreshes the
change cascade and SAR evidence for that AOI. No index rebuild. Measured live: searchable in 254 s.

## 6. False-alarm suppression

Cloud / shadow / haze → masks and `q` gating; season → harmonic baseline; single-date noise → persistence;
misregistration → shift term in `q`; sensor differences → per-item scale/offset and no cross-sensor pixel
differencing; SAR agreement as independent evidence. Precision is preferred: uncertain candidates stay stored
but hidden below the operating confidence.

## 7. Honest capability

At 10 m we detect construction sites and building clusters, land clearing, water-extent change and major roads.
We do not claim individual vehicles or small single buildings. The pipeline is resolution-agnostic (tiles in
metres), so finer organiser imagery flows through unchanged; metrics are reported per class and per GSD.

Design reference: `docs/DESIGN.md`; decisions: `docs/decisions/`; operating procedure: `docs/RUNBOOK.md`.
