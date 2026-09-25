# 0001 Data staging: AOI-window extracts, monthly series, sources

**Date:** 2026-09-25 · **Status:** accepted

## Context
The problem statement asks for every listed source. The laptop has 44 GB free, so full products (~1 GB per S2 tile per date) cannot be stored for 3 AOIs × ~90 months.

## Decision
- **AOIs:** navimumbai (construction/water/mangrove), dholera (road grid, new city, airport u/c), hasdeo (PEKB coal mine: forest clearance, seasonality). Together they cover all four change classes.
- **Series 2019-01 → 2026-08, one best acquisition per month:** AOI coverage ≥ 95% first, then lowest cloud (scene cloud < 40% for optical).
- **Raw = AOI-window extracts** of the provider COGs (original DN and dtype, resampled onto the AOI UTM grid). Each has sha256 + source href + STAC item + `scene.json`. Written atomically; never modified afterwards.
- **S2:** Earth Search `sentinel-2-c1-l2a`. Bands B02 B03 B04 B08 B8A B11 B12 + SCL + **CLD** (Sen2Cor cloud probability, for `CLOUD_PROB`). Scale/offset taken from `raster:bands`.
- **S1:** Planetary Computer `sentinel-1-rtc` (terrain-corrected γ⁰, CC-BY-4.0). **One relative orbit per AOI.** Stored as **int16 dB×100** (22 → 10 MB per band). Mosaic/resampling happens in linear first.
- **Landsat:** Planetary Computer `landsat-c2-l2`, L8/L9 only, native 30 m.
- **Labels/aux:** ESA WorldCover (2020, 2021), Impact Observatory annual LULC (stand-in for Dynamic World, which needs a GEE account), JRC GSW, Copernicus DEM, OSM roads yearly via ohsome, OSCD, EuroSAT RGB.
- **Models:** RemoteCLIP, GeoRSCLIP (ViT-B-32), RS-M-CLIP (M1 candidates), Prithvi-EO-2.0-300M-TL (eo_vec), TerraMind-1.0-base (SAR candidate, benchmark only).
- The staging CLI writes `manifests/models.yaml` and `manifests/datasets.yaml`. It stops when free disk < 8 GB. Resumable.

## Consequences
- "Every raw file has a SHA-256" holds for the extracts, not the original products. Provenance to the product goes through the source href + STAC item id.
- Monthly cadence limits earliest-date precision to about one month. Denser dates can be added later by incremental ingest.
- Not staged (accounts needed): Bhuvan/NRSC (mandatory), Dynamic World, NASA HLS.
