# DESIGN.md §6. Keep this file and that table in sync.
# AWS box / Linux: uv is on PATH. Windows laptop: make UV="python -m uv" <target> (needs GNU make).
UV      ?= uv
RUN     := $(UV) run
COMPOSE := docker compose --env-file docker/.env -f docker/compose.yaml
OFFLINE := $(COMPOSE) -f docker/compose.offline.yaml

.PHONY: up down up-offline offline-check test lint db-migrate stage ingest ingest-scene reconcile \
        audit-verify bench-clip eval report verify-weights stats

up:
	$(COMPOSE) up -d --build

down:
	$(COMPOSE) down

up-offline:            ## the proof run: app network has no external route
	$(OFFLINE) up -d
	$(MAKE) offline-check

offline-check:         ## /stats answers from inside; the internet does not
	$(OFFLINE) exec -T api python -c "import urllib.request as u; print(u.urlopen('http://localhost:8000/stats').status)"
	$(OFFLINE) exec -T api python -c "import urllib.request as u, sys; \
	  sys.exit('NETWORK REACHABLE - not offline') if u.urlopen('https://huggingface.co', timeout=5) else None" \
	  && exit 1 || echo "offline OK: no external route"

test:
	$(RUN) pytest -q

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .
	$(RUN) mypy governance        # add change query as they land (DESIGN.md §4: strict there)

db-migrate:
	$(COMPOSE) exec -T db sh -c 'psql -v ON_ERROR_STOP=1 -U "$$POSTGRES_USER" -d "$$POSTGRES_DB"' < index/schema.sql

stage:                 ## ONLINE, resumable
	$(RUN) python scripts/stage_data.py all

stats:
	$(RUN) python -c "from fastapi.testclient import TestClient; from api.main import app; import json; \
	  d = TestClient(app).get('/stats').json(); d.pop('staging_report', None); print(json.dumps(d, indent=1))"

# --- land with their milestones (DESIGN.md §13) ---
ingest:                ## make ingest AOI=<name>   (M2/M3)
	@echo "not implemented yet: worker ingest (M2)"; exit 1

ingest-scene:          ## make ingest-scene PATH=<file>   (M9)
	@echo "not implemented yet (M9)"; exit 1

reconcile:             ## rebuild Qdrant from PostGIS (M2)
	@echo "not implemented yet: index/reconcile.py (M2)"; exit 1

audit-verify:          ## (M8)
	@echo "not implemented yet: governance/audit.py (M8)"; exit 1

bench-clip:            ## M1 go/no-go part 1: zero-shot EuroSAT per candidate -> eval/reports/
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(RUN) python eval/bench_clip.py

eval:                  ## (M10)
	@echo "not implemented yet (M10)"; exit 1

report:                ## (M10)
	@echo "not implemented yet (M10)"; exit 1

verify-weights:
	$(RUN) python -c "import yaml, sys; from pathlib import Path; from ingest.validate import sha256; \
	  m = yaml.safe_load(open('manifests/models.yaml', encoding='utf-8')); \
	  bad = [f'{n}/{f}' for n, e in m.items() for f, h in e['sha256'].items() if sha256(Path(e['path']) / f) != h]; \
	  print('all weights verified' if not bad else f'MISMATCH: {bad}'); sys.exit(1 if bad else 0)"
