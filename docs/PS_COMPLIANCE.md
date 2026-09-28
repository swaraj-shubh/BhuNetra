# PS SIH26227: requirement-by-requirement compliance

Every clause of the official problem statement (`docs/PS_SIH26227.md`) mapped to what the system does, where it is,
and the evidence. Status: **Met** (built, demonstrated and measured), **Partial** (built, with a stated limit), **Gap**
(not done, with the reason). Numbers come from files in `eval/reports/` and `manifests/`; nothing is estimated.
Last checked against the running system: 2026-09-28.

## 2.2 "queryable by semantic content and by change over time, while retaining spatial, temporal and sensor filters"

| PS asks | What we do | Where | Evidence | Status |
|---|---|---|---|---|
| Support analyst **discovery**, not only known locations | Search the whole archive by words or by example; review queue of all detected changes, most confident first | Workbench tabs *Search* and *Review changes* | Live UI; `GET /change`, `POST /search/text` | Met |

### 2.2.1 Semantic and multimodal retrieval

| PS asks | What we do | Where | Evidence | Status |
|---|---|---|---|---|
| Free-text search over imagery tiles | Query parser (meaning, change intent and type, place, dates, sensor, land cover) + GeoRSCLIP text-image vectors per 640 m tile per season | `query/parser.py`, `query/retrieve.py`, `POST /search/text` | ~0.2-0.6 s per query; 352,475 tile-season vectors | Met |
| Rank-ordered results | Fusion score (similarity, change confidence, image quality, spatial fit, land-cover share) + duplicate removal | `query/fuse.py`, `query/dedup.py` | Unit tests; "why ranked" panel in UI | Met |
| **Image-to-image** search for similar locations | Prithvi-EO-2.0 vectors; seed = any place in the archive, **or an uploaded multiband GeoTIFF** | `POST /search/image {tile_id | chip}`, UI *Look-alikes* and *search with an image* | Uploaded tile finds itself at rank 2 (behind an overlapping neighbour), 1-2 s | Met |
| Refine by AOI, date range, sensor, other metadata | Area, from/to dates, sensor ("changes confirmed by Sentinel-1 / Landsat") | Search filters; `aoi`, `date_range`, `sensor` in the request | Live check: 2023-24 window returns only 2024 events | Met |
| "newly built structures near a river" | Parsed as change = construction, near water <= 1 km (hard limit) | Example query in UI | 20/20 results within 1 km of water | Met (water = river, lake, reservoir, creek or salt pan: the water map does not separate them) |
| "large vehicle concentrations on open ground" | Parsed; UI states vehicles are below 10 m resolution (weak signal) | `below_resolution` warning | Honest capability, DESIGN.md §16 | Partial by physics (10 m pixels) |
| Retrieval quality on held-out queries | Blind judging tool: 30 fixed queries x 10 results, shuffled, no scores | `/label/retrieval`, `eval/retrieval_pool.py` | 200-tile benchmark mAP 0.48 -> 0.53 with land cover (model-labelled); **human P@10 not yet judged** | Partial |

### 2.2.2 Multi-temporal change analysis

| PS asks | What we do | Where | Evidence | Status |
|---|---|---|---|---|
| For a **specified area** and time window | Area = any of the areas **or the area framed on the map** (`bbox`); time window from/to | `GET /change?aoi|bbox&t1&t2`, *Review changes* filters | Live check: bbox and dates filter correctly | Met |
| Appearance, disappearance, expansion, contraction | Direction stored per event, shown in plain words ("Water extent shrank") | `change/classify.py`, `change_events.direction` | Unit test per type | Met |
| Classify construction, clearance, water, road | Rule classifier on persistent index changes; unreliable "other" hidden by default | `change/classify.py` | Held-out type accuracy 0.44 (construction vs clearance confused at 10 m) | Partial |
| **Earliest** supporting observation with usable imagery | First observation of a persistent run of clear observations; the actual scene id, date and sensor are shown | `change/breaks.py`, evidence panel *When* | Earliest-date error median **0 days** on held-out | Met |

### 2.2.3 False-alarm suppression and quality handling

| Confounder named in the PS | How it is handled | Evidence | Status |
|---|---|---|---|
| Seasonal variation | Per-pixel harmonic seasonal model (annual + semi-annual + trend); a change must break it | `change/harmonic.py`; unit test: pure seasonality gives no break | Met |
| Cloud, haze, shadow | Sen2Cor SCL classes (cloud, shadow, cirrus) + cloud probability, dilated; observations with q < 0.6 never enter the series | `ingest/masks.py`, `ingest/quality.py` | Unit tests | Met |
| Snow | SCL snow masked per area (`mask_snow: true` in `configs/aoi/<name>.yaml`); Landsat QA snow always | `ingest/masks.py` | Unit test | Met (set the flag for a snowy organiser area) |
| Illumination / view angle | Sun-elevation term in q; seasonal composites of near-nadir Sentinel-2; persistence across many dates | `ingest/quality.py` | q formula test | Partial (no explicit BRDF correction) |
| Radiometric inconsistency | Per-item scale/offset incl. the 2022 processing-baseline offset; no cross-sensor pixel differencing | `ingest/stac.py`, DESIGN.md §9 | Verified on catalog data | Met |
| Imperfect co-registration | Sub-pixel phase correlation of every scene against a reference | `ingest/coreg.py`, `eval/reports/2026-09-28_coreg.json` | 205 scenes: median shift 0.08-0.18 px, 1 scene > 1 px; one bad scene cannot make an event (3 persistent observations needed) | Partial (measured; not yet used inside q) |
| Sensor differences | Independent cross-checks: Landsat 8/9, Sentinel-1 radar, annual land-cover map; each shown per event | `worker/landsat_evidence.py`, `sar_evidence.py`, `lulc_evidence.py` | Cross-sensor agreement shown in evidence panel | Met |
| Confidence estimates; precision over recall | Calibrated confidence (logistic + isotonic on our calibration split); stored-but-hidden below 0.55; unreliable type hidden | `change/calibrate.py`, `eval/fit_calibration.py` | **Blind held-out: precision 0.71, recall 0.84, F1 0.77**; false alarms on no-change places 0.67 -> 0.39 | Met |

### 2.2.4 Discovery and clustering

| PS asks | What we do | Where | Evidence | Status |
|---|---|---|---|---|
| Unsupervised grouping of similar sites across a wide area | UMAP + HDBSCAN on Prithvi vectors of the look-alikes of a chosen place; groups coloured on the map with a legend | `POST /discover/cluster`, UI *Groups* | ~1.5-3 s | Met |
| Find other locations from one site without a new query | *Look-alikes* from any result; overlapping neighbours excluded so results are genuinely elsewhere | `query/retrieve.py:overlapping` | Look-alikes 1.5 km+ away, across areas | Met |

### 2.2.5 Analyst workflow and provenance

| PS asks | What we do | Where | Status |
|---|---|---|---|
| Ranked review queue | Changes most confident first, filters, counts by type, keyboard J/K/C/R | *Review changes* | Met |
| Before-and-after evidence | Composites before/after with the acquisition dates they were built from; tile image under the changed outline on the map | Evidence panel | Met |
| Location | Coordinates and changed area in hectares | Evidence panel | Met |
| Acquisition time | Actual last-normal and first-changed observation dates with scene ids | Evidence panel *When* | Met |
| Sensor or source information | Sensors supporting the event; per-sensor cross-check verdicts | Evidence panel | Met |
| Confidence | High / medium / low (not shown as a probability: blind precision is 0.71) | Everywhere | Met |
| Relevant processing history | Audit-log entries of the run that created the event and every later step | Evidence panel *Processing history* | Met |
| Confirm / reject, preserved in the audit trail | Hash-chained audit log, verifiable | `POST /feedback`, `GET /audit/verify` (453 entries, intact) | Met |
| Feedback for reranking | Rocchio re-rank live on ✓ / ✗ | `POST /feedback` with `query_id` | Met |
| Exports keep source-scene and processing provenance | Ed25519-signed zip: GeoJSON + GeoPackage + provenance.json + manifest; signature verifies | `POST /export`, `POST /export/verify` | Met |

### 2.2.6 Scale, incremental ingestion and sovereignty

| PS asks | What we do | Evidence | Status |
|---|---|---|---|
| Efficient vector indexing | Qdrant (HNSW) rebuilt from PostGIS; index kept warm | Look-alike 0.15 s warm | Met |
| Incremental addition without a full rebuild | `POST /ingest` / `make ingest-scene`: only the affected season is recomputed | New scene searchable in 254 s | Met |
| Fully on-premises, no cloud or external APIs | Docker Compose; models from local paths; offline proof mode | `make up-offline`: containers have no internet route; full UI checked in this mode | Met |
| Preserve georeferencing and acquisition metadata | UTM grid in metres, per-file SHA-256, STAC items, lineage on every product | DESIGN.md §2.7 | Met |
| Ingest organiser GeoTIFF / COG | `make stage-local FILES=... AOI=...` (any GSD, band names inferred or given) | Rehearsal: 59 organiser-style GeoTIFFs staged in 35 s, searchable in 311 s | Met |

### 2.2.7 Constraints

| PS asks | What we do | Status |
|---|---|---|
| Demo with network disabled | Offline proof mode, and a plain `make up` on a disconnected machine | Met on the AWS box; **dress rehearsal on the actual demo machine still to do** |
| Public pretrained models, origin and licence declared, weights packaged | `manifests/models.yaml` (source, licence, version, SHA-256); weights verified before load | Partial: GeoRSCLIP licence is CC-BY-NC; written permission from its authors requested (fallback OpenAI CLIP, benchmarked) |
| Public or organiser imagery only | Sentinel-2, Sentinel-1, Landsat, JRC water, ESA WorldCover, IO land cover, OSM | Met |
| Primary sources incl. **Bhuvan** | Local-file ingest path ready | **Gap**: needs an NRSC account (team action) |

## 2.3 Expected solution and deliverables

| PS asks | Where | Status |
|---|---|---|
| Demo over an organiser area and time span | Area-agnostic pipeline; organiser-file rehearsal done | Met (rehearsed) |
| Retrieval evaluated on held-out queries | `eval/run.py` scores any queries file via the public API; our own blind judging pending | Partial |
| Change evaluated on held-out change / no-change cases | `eval/run.py` via `GET /change`, matched by geometry | Met: P 0.71 / R 0.84 / F1 0.77 on 121 blind cases |
| Source code | This repository | Met |
| Architecture note | `docs/ARCHITECTURE.md` | Met |
| Index-build and incremental-ingestion procedure | `docs/RUNBOOK.md` | Met |
| Model and dataset provenance | `manifests/models.yaml`, `manifests/datasets.yaml` | Met |
| Reproducible evaluation report: area, scenes/tiles, build time, storage, latency, hardware | `make report` -> `eval/reports/2026-09-28_report.md`; same figures live in the *System* tab | Met: 1,253 km², 772 scenes, 12,096 tiles per date, build 3 h 55 min, 4.4 GB database, query 0.1-0.6 s, T4 GPU |

## Open items, in priority order
1. Human retrieval judgements (`/label/retrieval`) -> retrieval P@10 in the report.
2. Blind calibration recheck (`/label` -> "Calibration recheck") -> refit calibration and the learned type classifier.
3. Offline dress rehearsal on the demo machine.
4. Bhuvan data (NRSC account) and the GeoRSCLIP licence answer (team).
5. Adopt the measured co-registration shift inside q (needs a rebuild and one more held-out evaluation).
