# DESIGN.md §6. Keep this file and that table in sync.
# AWS box / Linux: uv is on PATH. Windows laptop: make UV="python -m uv" <target> (needs GNU make).
UV      ?= uv
RUN     := $(UV) run
COMPOSE := docker compose --env-file docker/.env -f docker/compose.yaml
OFFLINE := $(COMPOSE) -f docker/compose.offline.yaml
# host-side tools (worker, reconcile) reach the stack via loopback ports; credentials from docker/.env
-include docker/.env
HOST_ENV := DATABASE_URL=postgresql://$(or $(POSTGRES_USER),sih):$(POSTGRES_PASSWORD)@127.0.0.1:5432/$(or $(POSTGRES_DB),sih)             QDRANT_URL=http://127.0.0.1:6333

.PHONY: up build change stage-local down up-offline offline-check test lint db-migrate stage ingest ingest-scene reconcile embed-eo sar landsat build-index \
        audit-verify bench-clip eval report verify-weights stats

up:                    ## start (no rebuild: exporting the CUDA image takes ~10 min even when cached)
	$(COMPOSE) up -d --wait

build:                 ## rebuild the api image after code/web changes (ONLINE)
	$(COMPOSE) build

down:
	$(COMPOSE) down

up-offline:            ## the proof run: app network has no external route
	$(OFFLINE) up -d --wait
	$(MAKE) offline-check

offline-check:         ## /stats answers from inside; the internet does not
	$(OFFLINE) exec -T api python -c "import urllib.request as u; print(u.urlopen('http://localhost:8000/stats').status)"
	$(OFFLINE) exec -T api python -c "import urllib.request as u; u.urlopen('https://huggingface.co', timeout=5)" \
	  && { echo "NETWORK REACHABLE - not offline"; exit 1; } || echo "offline OK: no external route"

test:
	$(RUN) pytest -q

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .
	$(RUN) mypy --strict change governance   # DESIGN.md §4 (add query when it lands)

db-migrate:            ## idempotent migrations on top of schema.sql (applied at initdb)
	for f in index/migrations/*.sql; do echo "== $$f"; 	  $(COMPOSE) exec -T db sh -c 'psql -v ON_ERROR_STOP=1 -U "$$POSTGRES_USER" -d "$$POSTGRES_DB"' < $$f || exit 1; done

stage:                 ## ONLINE, resumable
	$(RUN) python scripts/stage_data.py all

stats:
	$(RUN) python -c "from fastapi.testclient import TestClient; from api.main import app; import json; \
	  d = TestClient(app).get('/stats').json(); d.pop('staging_report', None); print(json.dumps(d, indent=1))"

# --- land with their milestones (DESIGN.md §13) ---
ingest:                ## make ingest AOI=<name>   (M2: S2 -> q -> composites -> text_vec)
	@$(HOST_ENV) $(RUN) python -m worker.ingest $(AOI)

stage-local:           ## make stage-local FILES="<files or folders>" AOI=<name> [MANIFEST=m.yaml]   (M3b, offline)
	$(RUN) python scripts/stage_local.py $(FILES) --aoi $(AOI) $(if $(MANIFEST),--manifest $(MANIFEST))

change:                ## make change AOI=<name>   (M3: indices -> harmonic breaks -> events)
	@$(HOST_ENV) $(RUN) python -m worker.change $(AOI)

ingest-scene:          ## make ingest-scene SCENE=<staged folder|GeoTIFF> AOI=<name>   (M9; SCENE, since PATH would clobber $$PATH)
	@$(HOST_ENV) $(RUN) python -m worker.jobs run --path $(SCENE) --aoi $(AOI)

embed-eo:              ## make embed-eo AOI=<name>   (M7: Prithvi eo_vec per seasonal composite, GPU if present)
	@$(HOST_ENV) $(RUN) python -m worker.embed_eo $(AOI)

sar:                   ## make sar AOI=<name>   (M9: Sentinel-1 log-ratio evidence on stored events)
	@$(HOST_ENV) $(RUN) python -m worker.sar_evidence $(AOI)

landsat:               ## make landsat AOI=<name>   (M9: Landsat 8/9 cross-sensor evidence on stored events)
	@$(HOST_ENV) $(RUN) python -m worker.landsat_evidence $(AOI)

build-index:           ## full index build for AOI (empty = all): ingest -> change -> eo_vec -> S1 evidence
	$(MAKE) ingest AOI=$(AOI) && $(MAKE) change AOI=$(AOI) && $(MAKE) embed-eo AOI=$(AOI) && $(MAKE) sar AOI=$(AOI) && $(MAKE) landsat AOI=$(AOI)

reconcile:             ## rebuild Qdrant from PostGIS (M2)
	@$(HOST_ENV) $(RUN) python -m index.reconcile

audit-verify:          ## recompute the audit hash chain (GET /audit/verify lands in M8)
	@$(HOST_ENV) $(RUN) python -m governance.audit

bench-clip:            ## M1 go/no-go part 1: zero-shot EuroSAT per candidate -> eval/reports/
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(RUN) python eval/bench_clip.py

eval:                  ## M10: score the running API (QUERIES=file.json CASES=file.geojson) -> eval/reports/
	$(RUN) python eval/run.py $(if $(QUERIES),--queries $(QUERIES)) $(if $(CASES),--cases $(CASES))

report:                ## M10: PS evaluation report (stats, build time, storage, latency, hardware, provenance, eval)
	$(RUN) python eval/report.py

verify-weights:        ## every model file vs manifests/models.yaml (governance/model_registry.py)
	$(RUN) python -c "import json, sys; from governance.model_registry import verify_all; r = verify_all(); print(json.dumps(r, indent=1)); sys.exit(any(v != 'ok' for v in r.values()))"
