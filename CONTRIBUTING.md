# Contributing

Read [`docs/DESIGN.md`](docs/DESIGN.md) first. Section 2 lists the invariants every change has to keep: heavy compute only at ingest, PostGIS before Qdrant, nothing reaches the network at runtime.

- One logical change per commit, message as `<module>: <imperative summary>`.
- Thresholds live in `configs/`, never in code. Tune only on the calibration split.
- Maths and governance code (`change/`, `query/fuse.py`, `query/rocchio.py`, `governance/`) gets tests first.
- Tests use synthetic rasters built in code; never real imagery.
- `make lint` and `make test` must pass before a pull request.
