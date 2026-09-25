"""Generic STAC staging (ONLINE, staging only): pick acquisitions, mosaic the AOI window, keep source DNs.

One code path for every source in configs/sources.yaml. Resumable: a date whose scene.json exists is skipped.
Output: data/raw/<source>/<aoi>/<date|year|static>/<BAND>.tif + scene.json; STAC items in data/stac/<source>/.
"""

import json
import logging
import math
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import planetary_computer
import rasterio
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from shapely.geometry import box, shape
from shapely.ops import unary_union

from configs.settings import aois, paths, sources, stac_url
from governance import lineage
from ingest import tiling
from ingest.validate import sha256

log = logging.getLogger(__name__)

GDAL_ENV = {
    "AWS_NO_SIGN_REQUEST": "YES",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff",
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "3",
    "GDAL_CACHEMAX": 256,
}
KEEP_PROPS = (
    "platform",
    "eo:cloud_cover",
    "sat:relative_orbit",
    "sat:orbit_state",
    "s2:processing_baseline",
    "landsat:wrs_path",
    "landsat:wrs_row",
    "view:sun_elevation",
    "view:sun_azimuth",
    "view:incidence_angle",
)


def load_aoi(name: str) -> tuple[dict, tuple[float, float, float, float]]:
    a = aois[name]
    return a.model_dump(), a.bbox


def aoi_names() -> list[str]:
    return list(aois)


DB_NODATA = -32768


def linear_to_db_int16(linear: np.ndarray) -> np.ndarray:
    """Linear power -> int16 dB*100. Non-positive/nodata -> DB_NODATA."""
    out = np.full(linear.shape, DB_NODATA, dtype=np.int16)
    ok = np.isfinite(linear) & (linear > 0)
    out[ok] = np.clip(np.round(1000 * np.log10(linear[ok])), -32767, 32767)
    return out


def band_meta(it, asset: str) -> tuple[float, float]:
    """(scale, offset) to apply to an asset's DNs.

    Earth Search `sentinel-2-l2a` items flagged `earthsearch:boa_offset_applied: true` already had the -1000 DN
    BOA_ADD_OFFSET removed, yet their raster:bands still say offset -0.1. Verified 2026-09-25: the same scene is
    exactly 1000 DN lower than in `sentinel-2-c1-l2a`. Applying -0.1 again would darken it by 0.1 (fake change).
    """
    rb = it.assets[asset].extra_fields.get("raster:bands", [{}])[0]
    scale, offset = rb.get("scale", 1.0), rb.get("offset", 0.0)
    if it.properties.get("earthsearch:boa_offset_applied"):
        offset = 0.0
    return scale, offset


def item_date(it) -> str:
    return (it.properties.get("datetime") or it.properties["start_datetime"])[:10]


def coverage(items, lonlat) -> float:
    """Fraction of the AOI bbox covered by the union of the items' footprints."""
    aoi = box(*lonlat)
    return unary_union([shape(i.geometry) for i in items]).intersection(aoi).area / aoi.area


def cloud(items) -> float:
    return max(i.properties.get("eo:cloud_cover", 0.0) for i in items)


def select_groups(src: dict, items, lonlat) -> dict[str, list]:
    """Map output key -> items to mosaic. month: best date per month; year: all tiles per year; once: all."""
    if src["every"] == "once":
        return {"static": items}
    if src["every"] == "year":
        by_year = defaultdict(list)
        for it in items:
            by_year[item_date(it)[:4]].append(it)
        return dict(by_year)

    if src.get("same_orbit"):
        # SAR geometry differs per orbit: keep the orbit with the most well-covered dates (DESIGN.md §5)
        dates = defaultdict(set)
        for it in items:
            if coverage([it], lonlat) >= 0.5:
                dates[(it.properties["sat:relative_orbit"], it.properties["sat:orbit_state"])].add(
                    item_date(it)
                )
        orbit = max(dates, key=lambda k: len(dates[k]))
        log.info(
            "  orbit %s %s (%d dates; others: %s)",
            *orbit,
            len(dates[orbit]),
            {f"{k[0]}{k[1][0]}": len(v) for k, v in dates.items() if k != orbit},
        )
        items = [
            i for i in items if (i.properties["sat:relative_orbit"], i.properties["sat:orbit_state"]) == orbit
        ]

    by_date = defaultdict(list)
    for it in items:
        by_date[item_date(it)].append(it)
    best = {}
    for date, its in by_date.items():
        cov = coverage(its, lonlat)
        if cov < sources["staging"]["min_aoi_coverage"]:
            continue
        month = date[:7]
        if month not in best or (cloud(its), date) < (cloud(best[month][1]), best[month][0]):
            best[month] = (date, its)
    return {date: its for date, its in best.values()}


def _stage_group(source: str, src: dict, aoi: str, key: str, items, epsg: int, bounds) -> dict:
    out_dir = paths.raw / source / aoi / key
    if (out_dir / "scene.json").exists():
        return {"key": key, "status": "exists"}
    tmp = out_dir.with_name(key + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)

    gsd = src["gsd"]
    width, height = math.ceil((bounds[2] - bounds[0]) / gsd), math.ceil((bounds[3] - bounds[1]) / gsd)
    transform = from_origin(bounds[0], bounds[3], gsd, gsd)
    signed = [planetary_computer.sign(i) for i in items] if src["stac"] == "planetary" else items

    bands = {}
    for asset, band in src["assets"].items():
        if not all(asset in it.assets for it in items):
            if asset in src.get("optional_assets", ()):
                continue  # e.g. CLD is absent in the sentinel-2-l2a fallback; masks work without it
            raise RuntimeError(f"asset {asset} missing")
        resampling = Resampling.nearest if band in src.get("nearest", ()) else Resampling.bilinear
        scale, offset = band_meta(items[0], asset)
        if any(band_meta(it, asset) != (scale, offset) for it in items[1:]):
            raise RuntimeError(f"mixed scale/offset for {band}; mosaic would be wrong")
        offset_metadata = items[0].assets[asset].extra_fields.get("raster:bands", [{}])[0].get("offset", 0.0)
        mosaic = nodata = None
        for it in signed:
            with rasterio.open(it.assets[asset].href) as ds:
                if mosaic is None:
                    nodata = (
                        ds.nodata
                        if ds.nodata is not None
                        else (np.nan if ds.dtypes[0].startswith("float") else 0)
                    )
                    mosaic = np.full((height, width), nodata, dtype=ds.dtypes[0])
                with WarpedVRT(
                    ds,
                    crs=f"EPSG:{epsg}",
                    transform=transform,
                    width=width,
                    height=height,
                    resampling=resampling,
                    nodata=nodata,
                ) as vrt:
                    arr = vrt.read(1)
            empty = np.isnan(mosaic) if np.isnan(nodata) else mosaic == nodata
            has = ~np.isnan(arr) if np.isnan(nodata) else arr != nodata
            mosaic[empty & has] = arr[empty & has]
        units = None
        if src.get("to_db"):
            # SAR: resampled/mosaicked in linear power above; stored as int16 dB*100 (halves disk, 0.01 dB steps)
            mosaic, nodata, units = linear_to_db_int16(mosaic), DB_NODATA, "dB"
            scale, offset = 0.01, 0.0
        path = tmp / f"{band}.tif"
        with rasterio.open(
            path,
            "w",
            driver="COG",
            width=width,
            height=height,
            count=1,
            dtype=mosaic.dtype,
            crs=f"EPSG:{epsg}",
            transform=transform,
            nodata=nodata,
            compress="deflate",
            predictor="YES",
        ) as dst:
            dst.write(mosaic, 1)
        empty = np.isnan(mosaic) if np.isnan(nodata) else mosaic == nodata
        bands[band] = {
            "file": path.name,
            "sha256": sha256(path),
            "scale": scale,
            "offset": offset,  # effective (see band_meta); raw catalog value in offset_metadata
            "offset_metadata": offset_metadata,
            "nodata": None if np.isnan(nodata) else nodata,
            "nodata_fraction": round(float(empty.mean()), 4),
            "sources": [it.assets[asset].href for it in items],
        } | ({"units": units, "to_linear": "10 ** (value * scale / 10)"} if units else {})

    stac_dir = paths.stac / source
    stac_dir.mkdir(parents=True, exist_ok=True)
    for it in items:  # unsigned copies: never persist SAS tokens
        (stac_dir / f"{it.id}.json").write_text(json.dumps(it.to_dict()))
    p = items[0].properties
    scene = {
        "aoi": aoi,
        "sensor": source,
        "date": key,
        "datetime": p.get("datetime") or p.get("start_datetime"),
        "platform": p.get("platform", source),
        "epsg": epsg,
        "bounds": list(bounds),
        "gsd": gsd,
        "collection": items[0].collection_id or src["collection"],
        "source_items": [it.id for it in items],
        "aoi_coverage": round(coverage(items, load_aoi(aoi)[1]), 4),
        "properties": {k: p[k] for k in KEEP_PROPS if k in p},
        "bands": bands,
    }
    (tmp / "scene.json").write_text(json.dumps(scene, indent=1))
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)
    return {"key": key, "status": "staged", "items": scene["source_items"]}


def stage_series(source: str, aoi: str, start: str, end: str, workers: int = 4) -> dict:
    src = sources["sources"][source]
    _, lonlat = load_aoi(aoi)
    client = Client.open(stac_url(source))
    items = _search(client, src["collection"], src, lonlat, start, end)
    if not items:
        raise RuntimeError(f"no {source} items for {aoi} {start}..{end}")
    groups = select_groups(src, items, lonlat)
    fallback = src.get("fallback_collection")
    if fallback and src["every"] == "month":
        # e.g. sentinel-2-c1-l2a lacks Jan-Nov 2022 for some tiles; fill only months the primary cannot
        missing = set(_gaps(src, groups, start, end))
        fb = [i for i in _search(client, fallback, src, lonlat, start, end) if item_date(i)[:7] in missing]
        extra = select_groups(src, fb, lonlat)
        groups.update(extra)
        log.info("  %s: filled %d of %d missing months from %s", source, len(extra), len(missing), fallback)
    epsg = tiling.aoi_epsg(lonlat)
    bounds = tiling.aoi_bounds_m(lonlat, epsg)
    log.info("%s/%s: %d items -> %d acquisitions to stage", source, aoi, len(items), len(groups))

    done, failed = [], []
    with rasterio.Env(**GDAL_ENV), ThreadPoolExecutor(workers) as pool:
        futs = {
            pool.submit(_stage_group, source, src, aoi, k, its, epsg, bounds): k for k, its in groups.items()
        }
        for f in as_completed(futs):
            key = futs[f]
            free_gb = shutil.disk_usage(paths.data).free / 1e9
            try:
                r = f.result()
                done.append(key)
                log.info(
                    "  %s/%s %s %s (%d/%d, %.1f GB free)",
                    source,
                    aoi,
                    key,
                    r["status"],
                    len(done),
                    len(groups),
                    free_gb,
                )
                if r["status"] == "staged":
                    lineage.append(
                        "staging",
                        lineage.record(
                            f"stage_{source}",
                            {"stac_items": r["items"]},
                            {"aoi": aoi, "key": key, "collection": src["collection"]},
                        ),
                    )
            except Exception as e:  # noqa: BLE001 - keep going; failures are listed and retried on the next run
                failed.append(key)
                log.warning("  %s/%s %s FAILED: %s", source, aoi, key, e)
            if free_gb < sources["staging"]["min_free_gb"]:
                log.error("Disk below %.0f GB free - stopping", sources["staging"]["min_free_gb"])
                for other in futs:
                    other.cancel()
                break
    return {
        "source": source,
        "aoi": aoi,
        "planned": len(groups),
        "done": len(done),
        "failed": sorted(failed),
        "months_without_acquisition": _gaps(src, groups, start, end),
    }


def _search(client, collection: str, src: dict, lonlat, start: str, end: str) -> list:
    query = dict(src.get("query", {}))
    if "max_cloud" in src:
        query["eo:cloud_cover"] = {"lt": src["max_cloud"]}
    search = client.search(
        collections=[collection],
        bbox=lonlat,
        query=query or None,
        datetime=None if src["every"] == "once" else f"{start}/{end}",
    )
    required = [a for a in src["assets"] if a not in src.get("optional_assets", ())]
    return [i for i in search.items() if all(a in i.assets for a in required)]


def _gaps(src: dict, groups: dict, start: str, end: str) -> list[str]:
    if src["every"] != "month":
        return []
    have = {k[:7] for k in groups}
    y, m = int(start[:4]), int(start[5:7])
    out = []
    while f"{y:04d}-{m:02d}" <= end[:7]:
        if f"{y:04d}-{m:02d}" not in have:
            out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def staged(source: str, aoi: str) -> list[Path]:
    return sorted((paths.raw / source / aoi).glob("*/scene.json"))
