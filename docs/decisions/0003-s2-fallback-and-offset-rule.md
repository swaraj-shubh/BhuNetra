# 0003 Sentinel-2: fallback collection for missing months + corrected offset rule

**Date:** 2026-09-25 · **Status:** accepted

## Context
- Dholera staging reported **no acquisitions Jan–Nov 2022**. Earth Search `sentinel-2-c1-l2a` has **zero items** for tiles 42QZK/43QBE in that period (its Collection-1 backfill is incomplete). The older `sentinel-2-l2a` has 12 per month.
- Radiometric check of the same scene (2022-12-28, 43QBE, red band): **C1 median DN 2210, old collection 1210**, exactly 1000 apart.
  - C1 items: `boa_offset_applied` absent and DNs carry +1000, so the metadata offset `-0.1` is **correct**.
  - Old collection, `boa_offset_applied: true`: the 1000 was already removed, yet `raster:bands` still says `-0.1`, a **stale value**. Applied naively, it darkens reflectance by 0.1 and fakes change.
  - Old collection, baseline 03.01 (before 2022-01-25): offset 0, correct. Baseline 04.00 with `applied: false`: `-0.1`, correct.

## Decision
- `configs/sources.yaml`: `fallback_collection: sentinel-2-l2a` for S2. It is used **only for months the primary cannot fill**.
- `ingest/stac.py:band_meta`: effective offset = catalog offset, **or 0 when `earthsearch:boa_offset_applied` is true**. Mixed-offset mosaics are checked on the effective value. `scene.json` records `offset` (effective), `offset_metadata` (raw) and the actual `collection`.
- `CLD` is optional (`optional_assets`); the fallback has no cloud-probability band, so masks use SCL only there.
- Unit test pins all three catalog states.

## Consequences
- 2022 is covered again. Fallback months come from original processing baselines (03.01/04.00), not the C1 reprocessing (05.x). A small radiometric difference is possible at the primary↔fallback boundary. **M3:** run a boundary check (a stable-target reflectance time series across 2021-12 → 2022-01 and 2022-11 → 2022-12) before trusting breaks there.
- Already-staged dholera C1 data is correct and is kept.
