# Runbook: index build, incremental ingestion, offline demo, evaluation

Linux host with Docker (Compose v2), NVIDIA driver + container toolkit for the GPU worker (optional: everything
runs on CPU, slower), `uv`, GNU make. Our reference box: AWS g4dn.xlarge (4 vCPU, 16 GB, Tesla T4), see
`docs/AWS.md`. All commands from the repo root.

## 0. Before the network goes off (online, once)

```bash
cp docker/.env.example docker/.env        # set POSTGRES_PASSWORD
uv sync                                   # host tools (worker CLI, eval)
make build                                # api/worker image incl. the web bundle (vendored assets, no CDN)
make stage                                # models + benchmarks + imagery for the configs/aoi/*.yaml AOIs (resumable)
make verify-weights                       # every model file vs manifests/models.yaml (sha256)
```

Staging writes `data/raw/<source>/<aoi>/<date>/` (AOI-window extracts + `scene.json` with per-band sha256) and
records every model / dataset in `manifests/`. Nothing else ever touches the network.

## 1. A new area of interest (e.g. the organisers' AOI)

1. `configs/aoi/<name>.yaml`: `name`, `type`, `why`, `bbox: [lon_min, lat_min, lon_max, lat_max]`.
   No thresholds per AOI (they are global in `configs/thresholds.yaml`).
2. Imagery:
   - public archive while online: `uv run python scripts/stage_data.py series --source all --aoi <name>`;
   - organiser files (GeoTIFF / COG, offline): `make stage-local FILES="/media/usb/*.tif" AOI=<name>
     [MANIFEST=m.yaml]`. The originals are only read; each file gets sha256 + `scene.json`
     (see `ingest/local.py` for what a manifest can supply: date, sensor, bands, scale/offset, nodata).
     Easiest: let `stage-local` create `configs/aoi/<name>.yaml` (bbox from the files). A hand-written AOI file
     needs `optical_source: organiser`, because `make ingest` reads only `data/raw/<optical_source>/<aoi>/`
     (default `sentinel2`); `stage-local` refuses with that hint otherwise.
     Example manifest (one multi-band file per date, band names in the TIFF): `sensor: generic`,
     `scale: 2.75e-05`, `offset: -0.2`, `nodata: 0`.

## 2. Full index build

```bash
make up                                   # db, qdrant, api, worker
make build-index AOI=<name>               # empty AOI = all AOIs
```

`build-index` = `make ingest` (validate → masks + q → tiles → seasonal composites → text_vec → PostGIS → Qdrant;
spatial context) → `make change` (indices → harmonic + persistent breaks → typed events) → `make embed-eo`
(Prithvi eo_vec, GPU) → `make sar` (Sentinel-1 evidence). Every step is safe to re-run (ingest skips scenes with a known sha256 and seasons already
built; eo_vec skips seasons with the current model version); per-stage timings go to `manifests/build.json`.

Measured on the reference box (g4dn.xlarge: 4 vCPU, 16 GB, T4; 3 AOIs, 1,253 km², 772 scenes), from an empty
database, 2026-09-27: ingest 1 h 56 min · change 53 min · eo_vec 57 min · S1 evidence 9 min · **total 3 h 56 min**
(Landsat evidence adds ~2 min). Upper bound: other jobs shared the box for most of the run. Output: 5,678 events and
352,475 vectors of each kind, identical to the live system (reproducible). Storage: ARD 8.0 GB, composites 8.6 GB,
PostGIS 4.2 GB, Qdrant 2.3 GB. Per-stage timings: `manifests/build_full_2026-09-27.json`.

## 3. Incremental ingestion (new acquisition, no rebuild)

```bash
# via the API (admin key when RBAC is on):
curl -X POST localhost:8000/ingest -H 'Content-Type: application/json' -H "X-API-Key: $KEY" \
     -d '{"path": "data/raw/sentinel2/<aoi>/<date>", "aoi": "<aoi>"}'      # -> {job_id}
curl localhost:8000/jobs/<job_id>                                         # status + timings
# or on the host:
make ingest-scene SCENE=data/raw/sentinel2/<aoi>/<date> AOI=<aoi>
```

The worker validates the scene, rebuilds only its season (composite + text_vec + eo_vec) → searchable, then
refreshes the AOI's change events and SAR evidence (analyst decisions are kept). Measured: searchable in 254 s,
change refresh complete in 1,029 s (dholera, 2026-09-23 scene).

## 4. Offline operation

- **Proof that nothing reaches the internet:** `make up-offline` (the app network is `internal: true`) then
  `make offline-check`: the api answers locally, an external URL fails from inside the api container, and the
  web UI answers on `http://localhost:8000` through `gate` (`docker/gate.py`, a stdlib relay run from the api
  image, the only container on a normal network). The whole workbench is usable in this mode (checked 2026-09-28:
  search, look-alikes, groups, review, export, system tab).
- **Demo:** disconnect the host from the network (unplug / disable the interface), then `make up` and open
  `http://localhost:8000`. The app loads models from local paths only (`HF_HUB_OFFLINE=1`), and the web bundle
  has no CDN assets or online basemaps.
- `make up` / `make up-offline` end by re-reading the vector index into the page cache (`make warm`, ~15 s):
  image builds evict it, and a cold look-alike query took 7-9 s instead of 0.15 s. Run `make warm` before a demo
  if anything heavy ran on the machine since.
- The api warms the clustering library at startup (numba JIT); "Groups" answers in ~2 s from the first click.

## 5. Evaluation and report

```bash
make eval QUERIES=queries.json CASES=cases.geojson   # -> eval/reports/<date>_eval.json
make report                                          # -> eval/reports/<date>_report.md
```

- `queries.json`: `{"queries": [{"id", "query", "aoi"?, "date_range"?, "relevant": [GeoJSON geometry, …]}]}`
- `cases.geojson`: features with `{"case_id", "label": "change" | "no_change", "t1", "t2", "type"?, "earliest_date"?}`
- Scoring uses only the public API (`/search/text`, `/change`) and matches by geometry (EPSG:4326), so hidden
  sets need no conversion. Our own labels convert with `eval/to_eval_format.py`.
- The report reads every number from measurements (`/stats`, the timed full build `manifests/build_full_*.json`
  as wall clock, the eval output, hardware, model / dataset manifests).
- The workbench **System** tab shows the same PS 2.3 figures live.

## 6. Integrity checks

```bash
make audit-verify          # recompute the audit hash chain (also GET /audit/verify)
make reconcile             # rebuild Qdrant from PostGIS (PostGIS is the source of truth)
make verify-weights        # model files vs manifests/models.yaml
make test && make lint
```
