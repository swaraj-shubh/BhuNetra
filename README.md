# Bhunetra

Search a satellite-imagery archive by meaning and by change over time, fully offline.

Built for Smart India Hackathon 2026, problem statement SIH26227 (Ministry of Defence).

Live demo: https://bhunetra.swarajz.xyz

## What it does

- **Text search**: "open-cast coal mine", "newly built structures near a river" returns ranked sites on the map.
- **Image search**: pick a tile or upload a GeoTIFF chip and find places that look like it.
- **Change detection**: construction, clearing, water-extent change and new roads, each with the earliest date it can be seen, a calibrated confidence and the scenes behind it.
- **Review**: confirm or reject results; the ranking updates live and every action goes into a hash-chained audit log.
- **Export**: Ed25519-signed GeoJSON / GeoPackage with full provenance.

Everything runs on one machine with the network disabled. Sentinel-2, Sentinel-1 and Landsat 8/9 are supported, and organiser-supplied GeoTIFF/COG files can be ingested at any resolution.

## Results

Three areas (Navi Mumbai, Dholera, Hasdeo), 1,253 km², 772 scenes, 2019–2026.

| | |
|---|---|
| Change detection, blind held-out set (121 cases) | precision 0.71, recall 0.84, F1 0.77 |
| Text search, 200 labelled tiles | mAP 0.53 |
| Text search latency (server) | ~140 ms |
| Full index build from empty | 3.9 h on one T4 |
| New scene searchable after upload | 254 s |

Full numbers, per class: [`eval/reports/2026-09-28_report.md`](eval/reports/2026-09-28_report.md).

## How it works

```
staging (online, once)   STAC → monthly AOI extracts + sha256
ingest                   masks → co-registration → quality score → 640 m tiles → seasonal composites
                         → GeoRSCLIP text_vec + Prithvi-EO-2.0 eo_vec
                         → harmonic baseline → persistent breaks → classifier → calibrated events
query                    parse → vector search (Qdrant) + events (PostGIS) → fuse → dedup → sites
```

PostGIS is the source of truth; Qdrant can be rebuilt from it at any time. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

**Stack:** Python 3.11, FastAPI, PostgreSQL 16 + PostGIS, Qdrant, PyTorch, GDAL/rasterio, React + MapLibre GL, Docker Compose.

## Running it

Needs Docker with Compose and about 60 GB of disk for data and models.

```bash
cp docker/.env.example docker/.env      # set POSTGRES_PASSWORD
make stage                              # download data and model weights (online)
make build
make up
make build-index                        # first run only
```

Open http://localhost:8000.

`make up-offline` starts the same stack with no route to the internet and checks that it really is cut off.

Development:

```bash
uv sync
make lint
make test
```

## Repository

```
api/          FastAPI app and web bundle
ingest/       scene validation, masks, co-registration, tiling, quality
features/     indices, composites, embeddings
change/       harmonic model, break detection, classification, calibration
query/        parser, retrieval, fusion, dedup, clustering, feedback
index/        PostGIS schema, Qdrant, reconcile
governance/   audit chain, signing, RBAC, model registry
worker/       job queue and batch pipelines
web/          React + MapLibre workbench
eval/         labels, benchmarks, evaluation and report
configs/      thresholds and per-area settings
docker/       Compose files and images
```

Docs: [design reference](docs/DESIGN.md) · [runbook](docs/RUNBOOK.md) · [deployment (AWS)](docs/AWS.md) · [deployment (GCP)](docs/GCP.md) · [problem statement](docs/PS_SIH26227.md) · [compliance matrix](docs/PS_COMPLIANCE.md) · [decisions](docs/decisions/)

## License

Code is MIT, see [LICENSE](LICENSE). Model weights and datasets keep their own licences, listed in [`manifests/`](manifests/).
