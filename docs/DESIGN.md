# Design reference: SIH26227 Satellite Semantic Retrieval & Change Analysis

## 1. What we are building (one paragraph)

An **offline, on-premises** system for the Indian Army (MoD / DGIS, SIH 2026 PS **SIH26227**). It makes a satellite-imagery archive (Sentinel-2, Landsat C2, Sentinel-1, Bhuvan) searchable **by meaning** (text or image queries) and **by change over time** (construction, clearance, water-extent change, road development, with the earliest supporting observation). It suppresses false alarms from cloud, season, shadow, misregistration and sensor differences. It gives analysts a ranked review queue with confirm/reject feedback, a tamper-evident audit trail, and signed exports with full provenance. The demo runs **with the network disabled**, on an **area chosen by the organisers**, and is scored on **hidden** retrieval queries and **hidden** change/no-change cases.

---

## 2. Architecture invariants (NON-NEGOTIABLE)

These rules define the architecture. Never break one silently. If a task seems to require breaking one, stop and raise it with the team.

1. **Heavy compute happens at ingest, never at query time.** Embeddings, the change cascade, ChangeFormer, classification and calibration all run in the `worker` during ingestion. The query path may only: encode the query text or chip, do vector/SQL lookups, fuse, dedup and cluster.
2. **Change events are stored with a fixed type and a fixed calibrated confidence.** Confidence changes only when a logged, versioned re-run of the cascade or calibration happens. Never recompute confidence inside a request.
3. **Two vectors, two jobs.** `text_vec` (remote-sensing CLIP) is used only for text→image search. `eo_vec` (Prithvi-EO-2.0) is used for image→image search, clustering and change evidence. Never compare a text embedding with `eo_vec`. Never build an "alignment" network between them.
4. **SAR never enters optical models.** Sentinel-1 feeds only the change cascade (log-ratio in dB, same relative orbit and pass direction).
5. **A change requires quality + alignment + persistence.** A single-date difference is never a change. Observations with `q < Q_MIN` never enter the time series.
6. **PostGIS is the source of truth; Qdrant is rebuildable.** Write order: PostGIS commit → Qdrant upsert. `index/reconcile.py` must always be able to rebuild Qdrant from PostGIS plus the stored vectors.
7. **Everything is traceable.** Every raw file has a SHA-256. Every derived raster has `lineage` JSON. Every event stores source scene IDs plus model/cascade versions. Every state-changing action writes a hash-chained `audit_log` entry.
8. **Offline at runtime.** No code path may reach the internet. Load models from local paths only, with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. No CDN assets in the web app. No online basemaps. The only code allowed to use the network is the staging CLI `scripts/stage_data.py`. It runs before the demo and is never part of the runtime.
9. **Tiles are defined in metres, not pixels.** Default tile 640 m, stride 320 m. Pixel size = metres / GSD. The pipeline must accept any GSD.
10. **Never tune on the organisers' hidden data.** Thresholds are tuned only on our calibration split. Final numbers come only from our frozen held-out split.
11. **Precision over recall.** When in doubt, a candidate stays hidden (stored, but below the operating threshold).
12. **Raw data is read-only.** Nothing ever writes to `data/raw/`. The one exception is the staging CLI, which creates new acquisition folders atomically (`<key>.partial` → rename) and never modifies an existing one.

---

## 3. The pipeline at a glance

```
STAGING (online, once: scripts/stage_data.py)                                  
STAC search → per-month best acquisition → AOI-window extract (COG) + sha256 + STAC item → data/raw/

INGEST TIME (worker)
raw scene → validate + SHA-256 + STAC item
→ standardise (AOI UTM grid, 10 m anchor, canonical band names)
→ masks (SCL + Sen2Cor CLD prob [s2cloudless only on L1C] | QA_PIXEL | S1 border) → AROSICS coreg (shift_px) → quality score q
→ tiles (640 m / 320 m stride) + COG + lineage
→ indices per valid observation (NDVI, MNDWI, NDBI, BSI) ; S1 log-ratio
→ seasonal composites → text_vec (CLIP) + eo_vec (Prithvi)
→ change cascade per dirty tile:
      harmonic fit → persistent break → evidence → [ChangeFormer check] → classifier → isotonic → event
→ PostGIS commit → Qdrant upsert → audit entries

QUERY TIME (api)
text  → parser {semantic, temporal, spatial, sensor}
image → eo_vec lookup/encode → similarity [+ HDBSCAN]
→ branches: semantic (Qdrant text_vec) | change events (SQL) | spatial/sensor filter (PostGIS)
→ fusion score → unique-site dedup → ranked sites

REVIEW
workbench → confirm/reject(+reason) → Rocchio (live rerank) ; rejected events → calibration pool
→ hash-chained audit log → Ed25519-signed GeoPackage/GeoJSON export
```

---

## 4. Tech stack (pin versions in `pyproject.toml` / `package.json`)

| Area | Choice |
|---|---|
| Language | Python 3.11 (backend, ML, pipeline), TypeScript (web) |
| Geo | GDAL, rasterio, rio-cogeo, pyproj, shapely, geopandas, pystac, arosics |
| Staging only | pystac-client, planetary-computer (anonymous SAS signing), huggingface_hub |
| ML | PyTorch (CPU wheels where there is no CUDA), open_clip / transformers, Prithvi-EO-2.0 weights, scikit-learn (isotonic; `sklearn.cluster.HDBSCAN` avoids the separate `hdbscan` build), lightgbm, umap-learn. s2cloudless only if L1C is staged (§9). |
| Stores | PostgreSQL 16 + PostGIS, Qdrant |
| API | FastAPI, pydantic v2, SQLAlchemy 2 + GeoAlchemy2 |
| Jobs | DB-backed queue or RQ + Redis (keep it simple) |
| Tiles to map | TiTiler (local) |
| Web | React + Vite + MapLibre GL (all assets bundled, no CDN) |
| Crypto | hashlib SHA-256, PyNaCl / cryptography Ed25519 |
| Packaging | Docker Compose; vendored wheels for offline builds; `uv` for Python envs and locking |
| Quality | ruff, mypy (strict on `change/`, `query/`, `governance/`), pytest |

Before adding a dependency, check that its licence is permissive and that it works offline, then add it to `manifests/` if it ships weights or data. Agree with the team before adding anything heavy (e.g. SNAP, GPU-only libraries).

---

## 5. Repository layout

```
sih26227/
├── docs/
│   ├── DESIGN.md             # this file
│   └── decisions/            # ADRs: NNNN-short-title.md
├── docker/                   # Dockerfiles, compose.yaml, offline build scripts
├── configs/                  # aoi/*.yaml, bands.yaml, thresholds.yaml, fusion.yaml, models.yaml
├── manifests/                # datasets.yaml, models.yaml (licence+sha256), build.json
├── ingest/                   # validate, stac, standardise, masks, coreg, tiling, quality
├── features/                 # indices, composites, embed_clip, embed_prithvi, sar
├── change/                   # harmonic, breaks, evidence, validator, classify, calibrate, events
├── index/                    # db (models), schema.sql / alembic, qdrant, reconcile
├── query/                    # parser, retrieve, fuse, dedup, cluster, rocchio
├── governance/               # audit (hash chain), sign, rbac, model_registry
├── api/                      # FastAPI app, routers, schemas
├── worker/                   # job runner, ingest orchestration
├── web/                      # React + MapLibre workbench
├── eval/                     # labels/, run_queries, run_change_cases, ablations, report
├── scripts/                  # one-off CLIs (download staging, benchmarks)
├── tests/                    # unit/, integration/, fixtures/ (tiny synthetic rasters)
└── data/                     # NOT in git (see §9)
```

Module boundaries: `ingest` and `features` never import `api`. `query` never imports `change` internals; it reads stored events through `index`. `governance` has no ML dependencies.


---

## 6. Commands

These are defined in the `Makefile` (create them in M0 if missing; keep this table in sync):

| Command | Does |
|---|---|
| `make up` / `make down` | Start/stop the Compose stack (db, qdrant, api, worker, tiles, web) |
| `make build` | Rebuild the api image (with the web bundle) after code changes. Online. `make up` no longer rebuilds |
| `make up-offline` | Start with external networking disabled (the proof run). The web UI stays usable through the `gate` relay (`docker/gate.py`) |
| `make warm` | Re-read the vector index into the page cache (run by `make up`; builds evict it, cold look-alike query 7-9 s vs 0.15 s) |
| `make test` | `pytest -q` (unit + integration on synthetic fixtures) |
| `make lint` | `ruff check . && ruff format --check . && mypy change query governance` |
| `make db-migrate` | Apply schema migrations |
| `make stage` | `python scripts/stage_data.py all` (online; resumable) |
| `make ingest AOI=<name>` | Ingest all staged scenes for an AOI |
| `make ingest-scene SCENE=<file> AOI=<a>` | Incremental ingest of one scene (`SCENE`, since `PATH` would clobber the shell PATH); same as `POST /ingest` |
| `make build-index AOI=<a>` | Full index build: `ingest` → `change` → `embed-eo` → `sar` → `landsat` → `calibrate` (empty AOI = all) |
| `make reconcile` | Rebuild/repair Qdrant from PostGIS |
| `make audit-verify` | Recompute the audit hash chain |
| `make bench-clip` | M1 go/no-go CLIP benchmark (EuroSAT + own tiles) |
| `make eval` | Run retrieval + change evaluation on the held-out split |
| `make report` | Generate the PS-required evaluation report |
| `make verify-weights` | Check every model file against `manifests/models.yaml` |

**Before claiming a task is done:** run `make lint` and `make test`, and paste the relevant output.

**Compute:** heavy work (staging, benchmarks, the Compose stack) runs on a Linux box; see `docs/AWS.md`.


---

## 7. Canonical definitions (use these exact names everywhere)

**Identifiers**
- `tile_id` = `<aoi>_<utmzone>_<row:04d>_<col:04d>` on the fixed AOI grid (same place = same ID across dates and sensors). Anchor the AOI grid origin at a multiple of `STRIDE_M` in UTM coordinates, so tiles from overlapping AOIs share footprints. Implemented in `ingest/tiling.py` (row 0 = northernmost). `tests/unit/test_ingest.py` enforces ids and footprint alignment.
- `scene_id` = original product ID (e.g. S2 product name); `sha256` = hex digest of the raw file
- `event_id` = `CHG-<6-digit>`; `obs_id`, `comp_id` = UUIDv7

**Canonical bands**: `blue, green, red, nir, nir_narrow, swir1, swir2` (+ `vv`, `vh` for S1). The mapping from each sensor's band names is in `configs/bands.yaml`.

**Indices**
```
NDVI  = (nir - red) / (nir + red)
MNDWI = (green - swir1) / (green + swir1)
NDBI  = (swir1 - nir) / (swir1 + nir)
BSI   = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue))
```

**Quality score** (per tile × observation)
```
q = valid_frac * clip(1 - shift_px, 0, 1) * (1 - 0.5*cloud_near) * (1 - 0.3*[sun_elev < 25])
```
`sun_elev` comes from each acquisition's `scene.json → properties["view:sun_elevation"]` (staged for S2 and Landsat).

**Harmonic model** (fit on the baseline period; t in days)
```
ŷ(t) = a0 + a1 cos(2πt/365.25) + b1 sin(2πt/365.25) + a2 cos(4πt/365.25) + b2 sin(4πt/365.25) + c·t
r(t) = (y - ŷ) / σ,   σ = 1.4826 * MAD(baseline residuals)
break = first obs with |r| > R_MAX followed by K_PERSIST-1 more same-sign exceedances
earliest_supported_date = first valid obs of the persistent run; last_normal_date = last valid obs before it
```

**Fusion score**
```
score = w_sim*sim_norm + w_conf*event_conf + w_q*q + w_sp*spatial_fit     (weights in configs/fusion.yaml)
```

**Rocchio**: `q' = normalise(α q + β mean(confirmed) − γ mean(rejected))`, α=1.0, β=0.75, γ=0.15

**Audit hash**: `entry_hash = sha256(prev_hash ‖ ts ‖ actor ‖ action ‖ canonical_json(payload))`, genesis `prev_hash = "0"*64`

**Change types (enum)**: `construction | clearance | water | road | other`; **direction**: `appearance | disappearance | expansion | contraction`

**Encoder inputs.** A 640 m tile at 10 m is only 64 px, but both encoders expect 224 px (**# VERIFY** on each model card):
- `text_vec`: resample the 64 px composite chip to the CLIP input size. The M1 benchmark must use the same resampling it will be deployed with.
- `eo_vec`: run Prithvi on 224 px (2240 m) context windows and average the patch tokens that fall inside each tile (16 px patch = 160 m, so 4×4 patches per tile). This costs one forward pass per window, not per tile. Proposal, **# VERIFY** against Prithvi's patch size and positional embedding.

---

## 8. Configuration and thresholds

- **No magic numbers in code.** Every threshold lives in `configs/thresholds.yaml` (created with the values below) and is loaded through a typed pydantic settings object. Initial values:

```yaml
quality:   { Q_MIN: 0.6, SHIFT_ACCEPT_PX: 0.5, SHIFT_REJECT_PX: 1.0, VALID_FRAC_IDX: 0.8, VALID_FRAC_EMB: 0.95, CLOUD_PROB: 0.4, CLOUD_DILATE_PX: 3 }
tiling:    { TILE_M: 640, STRIDE_M: 320 }
breaks:    { R_MAX: 3.0, K_PERSIST: 3, BASELINE_MONTHS: 12 }
classify:  { CONS_DNDVI: -0.15, CONS_DNDBI: 0.08, CLR_DNDVI: -0.20, CLR_DBSI: 0.08, WATER_DMNDWI: 0.25, WATER_FRAC: 0.10, ROAD_ELONG: 4.0, ROAD_WIDTH_PX: 3 }
events:    { OPERATING_CONF: 0.5, TARGET_PRECISION: 0.85 }
retrieval: { TOP_N: 500, TOP_K_SITES: 20, DEDUP_RADIUS_M: 320 }
cluster:   { MIN_CLUSTER_SIZE: 10, UMAP_DIM: 20 }
```

- A change to a threshold is a **decision**: record it in the commit message (what, why, what evidence) and bump `cascade_version` if it affects events.
- Every AOI-specific setting goes in `configs/aoi/<name>.yaml`. **No per-AOI hand-tuning of thresholds.** The organisers' AOI is unseen. Dev AOIs: `configs/aoi/{navimumbai,dholera,hasdeo}.yaml`.
- `CLOUD_PROB` applies to the Sen2Cor cloud-probability band `CLD` (0–100, so the threshold is 40) unless s2cloudless on L1C is adopted (§18).

---

## 9. Data rules

- `data/` is git-ignored. Layout: `raw/` (read-only) · `ard/` · `composites/` · `stac/` · `labels/` · `osm/` · `manifests/` · `benchmarks/` (EuroSAT, OSCD).
- Every downloaded dataset goes into `manifests/datasets.yaml` (source, licence, date, sha256). Every model goes into `manifests/models.yaml` (source URL, licence, version, sha256, role). **The staging CLI writes both automatically.**
- **What "raw" means here:** a full product per date (about 1 GB per S2 tile) does not fit on disk. `data/raw/<source>/<aoi>/<YYYY-MM-DD | YYYY | static>/` therefore holds **AOI-window extracts** of the provider's COGs. They keep the original digital numbers and dtype and are resampled onto the AOI UTM grid. Each folder has `scene.json`: per-band sha256 of the extract, the source hrefs, scale/offset, nodata, STAC item ids, AOI coverage, cloud/orbit/sun properties. Full STAC items are in `data/stac/<source>/`. `scene_id` = the STAC item id(s).
- **Sentinel-2 L2A:** handle the processing-baseline ≥ 04.00 `BOA_ADD_OFFSET` before scaling reflectance. Missing it creates fake change in January 2022. Apply the **effective** offset from `ingest/stac.py:band_meta`, not the raw catalog value: `raster:bands` scale/offset, **except that items flagged `earthsearch:boa_offset_applied: true` get offset 0**. Earth Search already removed the 1000 DN from those, but the metadata still says `-0.1` (the same scene is exactly 1000 DN lower than in C1). `sentinel-2-c1-l2a` (primary) = `0.0001`/`-0.1`, correct. It **lacks Jan–Nov 2022** for some tiles, so those months fall back to `sentinel-2-l2a` (ADR 0003). `scene.json` stores both the effective `offset` and the raw `offset_metadata`. Never hardcode by date. Staging refuses to mosaic items with mixed scale/offset. SCL invalid classes: 0, 1, 3, 8, 9, 10 (11 snow per AOI).
- **Landsat C2 L2:** apply scale 0.0000275 and offset −0.2; mask QA_PIXEL (fill, dilated cloud, cirrus, cloud, shadow, snow). QA_PIXEL nodata = 1 (the fill bit). Staged from Planetary Computer (the USGS AWS bucket is requester-pays). Landsat 8/9 only (no L7 SLC-off). Native 30 m grid with the same origin.
- **Sentinel-1:** σ⁰ → speckle filter → terrain correction → dB; compare only the same relative orbit and pass direction. The staged product is Planetary Computer `sentinel-1-rtc`: terrain-flattened **γ⁰** with terrain correction already applied. It is stored as **int16 dB×100** (mosaic/resampling done in linear power first; `linear = 10**(v/1000)`). This halves the disk use. Speckle-filter at ingest in linear. Staging keeps **one relative orbit per AOI** (the one with the most dates).
- Mask resampling uses nearest-neighbour; reflectance resampling uses bilinear.
- **Tests never need real imagery.** Use tiny synthetic GeoTIFFs in `tests/fixtures/` (build them in code with a known CRS and transform).
- **Labels:** `data/labels/` holds our query-relevance, change/no-change and earliest-date sets. The held-out split (30%, split **by AOI**) is frozen. Never read it outside `eval/`. See §18 for the AOI-count conflict.
- **OSM roads:** yearly snapshots (2019–2026) come from **Overpass attic queries** (`[date:…]`, full geometry) → `data/raw/osm/<aoi>/roads_<date>.geojson`. The ohsome `/elements/geometry` endpoint returns HTTP 403.
- **Staged inventory and gaps** (months without a clear acquisition, failures) are listed in `data/manifests/staging_report.json`.

---

## 10. Coding conventions

- Type hints everywhere; pydantic models at every API and config boundary; dataclasses internally.
- Pure functions for the maths (`change/harmonic.py`, `query/fuse.py`, `governance/audit.py`) with unit tests. Keep I/O at the edges.
- Every raster function takes and returns georeferenced data (array + transform + CRS, or a rasterio dataset). **Never drop the georeference.**
- Use structured logging (JSON) with `scene_id` / `tile_id` / `event_id` in context. No `print`.
- Wrap each long ingest stage in a timing context that writes to `manifests/build.json` (these numbers go into the report).
- Every DB write that changes state also writes an audit entry, in the **same transaction**.
- Names: snake_case modules; `*_m` for metres, `*_px` for pixels, `*_db` for decibels, `_date` for dates, `_ts` for timestamps (UTC, ISO-8601).
- Error handling: a bad scene is rejected with a reason stored in the DB. It must never crash the batch.
- Idempotency: re-ingesting the same file (same sha256) is a no-op.
- Write text files as UTF-8 explicitly (`encoding="utf-8"`). The Windows default codepage corrupts non-ASCII. Never run formatters on `models/`, `data/` or `web/vendor/` (they are third-party; ruff excludes them).

---

## 11. Testing requirements

| Area | Must have |
|---|---|
| `change/harmonic` + `breaks` | Synthetic seasonal series: pure seasonality → **no** break; step change → break at the right date; single spike → **no** break; cloudy obs (q<Q_MIN) ignored |
| `change/classify` | One synthetic case per change type + "other" |
| `ingest/tiling` | tile_id stable across dates; tile size correct at 10 m and 1 m GSD |
| `ingest/quality` | q formula, mask dilation |
| `query/parser` | Table-driven tests including the PS examples: "newly built structures near a river", "large vehicle concentrations on open ground" |
| `query/fuse` + `dedup` | Deterministic ranking; overlapping tiles collapse to one site |
| `query/rocchio` | Confirmed vector pulls toward; rejected pushes away |
| `governance/audit` | Chain verifies; editing any row breaks verification at that seq |
| `governance/sign` | Export signature verifies; tampered file fails |
| Integration | Synthetic AOI end-to-end: ingest → event exists → `/search/text` + `/change` return it |
| Offline | CI job runs the integration test with the network disabled |

Existing: `tests/unit/test_ingest.py` (S2 scale/offset, SCL+CLD mask with dilation, tile size at 10/1/30 m, tile_id stability and shared footprints, S1 dB round-trip, strict threshold config). Also `test_quality.py` (q formula, valid_frac/cloud_near, sha256-checked scene load), `test_composites.py` (IMD seasons, median, q gating), `test_audit.py` (chain verifies; editing/deleting a row breaks it at that seq), and **`tests/integration/test_pipeline.py`** (synthetic AOI → ingest → PostGIS/Qdrant → `/search/text`, idempotent rerun, reconcile, network blocked; needs `TEST_DATABASE_URL` to a throwaway `*test*` DB, provided in CI).

---

## 12. API contract (keep stable; the evaluation scripts depend on it)

```
POST /ingest                 {path}                               -> {job_id}
GET  /jobs/{id}                                                   -> status, timings
POST /search/text            {query, aoi?, date_range?, sensor?, k?} -> {parsed, sites[]}
POST /search/image           {chip | tile_id, filters?, k?}      -> {sites[]}
POST /discover/cluster       {seed_tile_id | aoi, params?}       -> {clusters[]}
GET  /change                 ?aoi&t1&t2&type&min_conf            -> {events[]}      # evaluation interface
GET  /events/{id}                                                -> full evidence + provenance
POST /feedback               {target, decision, reason, query_id} -> {reranked sites[]}
POST /export                 {event_ids | query_id, format}      -> signed package
GET  /audit/verify                                               -> {ok, broken_at?}
GET  /stats                                                      -> area, scenes, tiles, storage, latency
```
Every `site` and `event` in a response includes `provenance: {scene_ids, model_versions, cascade_version}`.
Implemented so far: `GET /stats`, `POST /search/text` (parser + fusion, returns `query_id`), `GET /change`, `GET /events/{id}`, `POST /feedback` (Rocchio), `POST /export` (Ed25519-signed zip: geojson, `format: gpkg` adds `export.gpkg` via a stdlib GeoPackage writer `api/gpkg.py`), `GET /audit/verify`; extras `POST /export/verify`, `GET /chips/{tile_id}/{season}.png`, `/label*` (M5 labelling). `POST /search/image` also takes `chip` (base64 multiband GeoTIFF, bands named blue..swir2; encoded by Prithvi at query time); `GET /chips/{tile_id}/{season}.tif` returns the 6-band composite. `GET /change` also takes optional `bbox` (min_lon,min_lat,max_lon,max_lat), `unreviewed`, `limit` (workbench paging; response adds `total`) and `all_types` (types in `events.HIDDEN_TYPES`, today `other`, are hidden unless asked). `GET /events/{id}` adds `location`, `area_ha`, `sensors`, `observations` (actual last-normal / first-changed scenes), `images` (before/after composites with acquisition dates) and `history` (audit entries). **Auth (M8):** `X-API-Key` → `governance/rbac.py` roles viewer < analyst < admin, enforced when `data/keys/users.yaml` exists (`python -m governance.rbac add <name> <role>`); chips stay unauthenticated (loaded by `<img>`).

---

## 13. Build order and status

Always work on the **earliest unfinished milestone** unless the team says otherwise.

| # | Milestone | Exit criterion (must be demonstrable) |
|---|---|---|
| M0 | Skeleton: repo, Compose, configs, manifests, Makefile, schema | `make up-offline` works; `/stats` responds; CI green |
| M1 | **CLIP go/no-go benchmark** | Table: zero-shot EuroSAT + P@10 on 200 own tiles for each candidate → model chosen, recorded as an ADR |
| M2 | First search: 1 S2 AOI → tiles → composites → text_vec → Qdrant → map | Typed query shows ranked tiles on the map |
| M3 | Time series: all S2 dates → q → indices → harmonic + breaks → events (rules) | Timeline with breakpoint for a known site; OSCD sanity check |
| M4 | Query parser + change branch + spatial filter + fusion + dedup | "newly built structures near a river" returns construction events near water |
| M5 | Labelled sets + threshold tuning + isotonic calibration | P/R/F1 on held-out split; reliability diagram |
| M6 | Workbench: queue, before/after, timeline, provenance, confirm/reject, Rocchio | Rejecting 2 results changes the ranking live |
| M7 | eo_vec (Prithvi) + image query + HDBSCAN | Click a site and see look-alike cluster |
| M8 | Governance: audit chain, signed export, RBAC, weight verification | Tampering detected; signature verifies |
| M9 | Incremental ingest; Landsat; S1 log-ratio; ChangeFormer (only if ablation helps) | New scene added live with no rebuild; ablation table |
| M10 | Eval harness, report generator, unseen-AOI rehearsal, offline dress rehearsal | Report auto-generated; full demo with network off |

**Scope rule:** nothing from M9 starts before M8 is done, unless the team explicitly asks.

---

## 14. Working conventions

1. **Before a change:** read the relevant section here and the module, and name the milestone it belongs to.
2. **Plan before coding** for anything touching more than one module: list files to change, the invariants involved (§2), and how you will test it. Get a review before large or architectural changes.
3. **Small, reviewable steps.** One logical change per commit. Commit message format: `<module>: <imperative summary>` (e.g. `change: add persistence check to break detector`).
4. **Tests first for maths and governance code** (harmonic, breaks, fuse, rocchio, audit, sign).
5. **Never invent numbers.** If a benchmark or metric is needed, run it or say it has not been measured. Placeholder values must be marked `# TODO(measure)`.
6. **Never fabricate model/dataset facts** (licence, band order, input size, embedding dim). Read the model card or config in `manifests/`, or ask. Mark unknowns `# VERIFY`.
7. **Record decisions.** Architectural choices go into `docs/decisions/NNNN-title.md` (context, decision, consequences). Threshold changes go into the commit message.
8. **Performance awareness.** Anything that runs per tile × date must be vectorised (numpy/xarray), not a Python loop over pixels. Report timings.
9. **After a change:** list any `# VERIFY` or `# TODO(measure)` items you added in the pull request.
10. If a request conflicts with §2, say so plainly and propose an alternative that keeps the invariant.

---

## 15. Things we never do

- Run ChangeFormer, the classifier or calibration inside an API request.
- Treat a single-date difference, or a cloudy observation, as change evidence.
- Feed VV/VH into CLIP or Prithvi; claim RGB CLIP understands multispectral bands.
- Claim vehicle-level detection on 10–30 m imagery (see §16).
- Pretrain or fully fine-tune a foundation model.
- Hard-code thresholds, paths, AOIs or model names in code.
- Call external APIs, CDNs, online tile servers or `from_pretrained("<hub-id>")` without a local path.
- Write to `data/raw/` (outside the staging CLI), or modify an `audit_log` row.
- Tune on, look at, or cache the organisers' hidden evaluation data.
- Report metrics computed on hand-picked examples.

---

## 16. Honest-capability reminder (for UI text, docs and the report)

At 10 m (Sentinel-2) we **can** detect construction sites and building clusters, land clearing, water-extent change and major roads. We **cannot** reliably detect individual vehicles or small single buildings. Dense vehicle concentrations give only a weak, low-confidence signal. The pipeline is resolution-agnostic, so finer organiser-supplied imagery flows through unchanged. UI copy and reports must reflect this; report metrics **per class and per GSD**.

---

## 17. Glossary

GSD (ground size of a pixel) · ARD (analysis-ready data) · COG (Cloud Optimized GeoTIFF) · STAC (catalog standard) · q (per-observation quality score) · harmonic model (seasonal baseline) · persistence (break must last K valid obs) · isotonic calibration (score → probability) · Rocchio (feedback query update) · HDBSCAN (density clustering, unknown k) · AROSICS (sub-pixel co-registration) · σ⁰ / γ⁰ (SAR backscatter; γ⁰ = terrain-flattened) · RTC (radiometric terrain correction) · ADR (architecture decision record)

---

## 18. Open questions

| # | Conflict | Proposed resolution |
|---|---|---|
| C2 | "Every raw file has a SHA-256" assumes full products; they don't fit on the 44 GB free. | Accept AOI-window extracts with per-file sha256 + source href + STAC item (§9). ADR 0001. |
| C3 | s2cloudless is trained on **L1C** TOA, but we stage **L2A**. | Use Sen2Cor `CLD` + SCL (already staged). Stage L1C and run s2cloudless only if the M3 ablation shows missed clouds. |
| C4 | S1 rule says σ⁰ + terrain correction; the staged RTC is already terrain-corrected γ⁰, stored as dB×100 int16. | Accept (§9). ADR 0001. |
| C5 | 640 m tiles = 64 px, but encoders take 224 px. | Encoder-input rule in §7 (upsample for CLIP; context window + token pooling for Prithvi). M1 must benchmark exactly this. |
| C6 | Held-out = 30% **by AOI**, but we have 3 dev AOIs, so the held-out set is 1 AOI. | Add 2–3 more AOIs when disk allows, or split by ≥10 km blocks within AOIs. |
| C8 | Bhuvan is a mandatory source and needs an NRSC login. | A team member registers and downloads into `data/raw/bhuvan/`. A local-file staging path (sha256 + `scene.json` for manually downloaded products) still has to be written. **Not implemented yet.** |
