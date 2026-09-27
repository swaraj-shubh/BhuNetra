"""Stage local / organiser GeoTIFF or COG files into the standard raw layout (M3b; PS §2.2.6, §2.2.7).

Staging, not runtime: reads the organiser's files (never modifies them), records their sha256, and writes
data/raw/<source>/<aoi>/<date>/<CANON>.tif + scene.json on the AOI grid, keeping the original DNs and dtype,
exactly like the STAC staging. From there the normal worker pipeline runs unchanged.

Nothing is guessed silently. Date, band mapping and scale must come from the files (name, TIFF tags, band
descriptions) or from a --manifest; otherwise the file is rejected with the missing piece named.
Manifest keys (all optional): date, sensor, bands {canonical: band index (1-based, multi-band file) | file-name
token (one file per band)}, scale, offset, nodata. No cloud mask is read from local files yet: scene.json says
`cloud_mask: none` and the persistence rule is the cloud defence (§2.5).
"""

import json
import math
import re
import shutil
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import yaml
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform, transform_bounds

from configs.settings import T, bands, paths
from ingest import tiling
from ingest.validate import sha256

__all__ = ["LocalStageError", "canonical_for", "detect_date", "sensor_family", "sha256", "stage"]

CANON = list(bands["sentinel2"]["bands"])  # canonical band names (DESIGN.md §7)
_DATE_RES = (
    re.compile(
        r"(?<!\d)((?:19|20)\d{2})[-_]?([01]\d)[-_]?([0-3]\d)(?!\d)"
    ),  # 20240315, 2024-03-15, 2024_03_15
)
_TAG_DATE = re.compile(r"((?:19|20)\d{2})[:\-/]([01]\d)[:\-/]([0-3]\d)")
_DATE_TAGS = ("ACQUISITION_DATE", "DATE_ACQUIRED", "acquisition_date", "datetime", "TIFFTAG_DATETIME")


class LocalStageError(ValueError):
    """A file cannot be staged without more information (the message names what is missing)."""


def _valid(y: str, m: str, d: str) -> str | None:
    try:
        return date(int(y), int(m), int(d)).isoformat()
    except ValueError:
        return None


def detect_date(name: str, tags: dict[str, str]) -> str | None:
    for rx in _DATE_RES:
        for m in rx.finditer(name):
            if found := _valid(*m.groups()):
                return found
    for k in _DATE_TAGS:
        if k in tags and (m := _TAG_DATE.search(tags[k])) and (found := _valid(*m.groups())):
            return found
    return None


def sensor_family(name: str, tags: dict[str, str]) -> str | None:
    hay = (name + " " + " ".join(tags.values())).upper()
    for fam, hints in bands["local_sensor_hints"].items():
        if any(h.upper() in hay for h in hints):
            return str(fam)
    return None


def canonical_for(token: str, family: str | None) -> str | None:
    """Canonical band for a band description / file token; sensor-specific tokens only with a known family."""
    t = token.strip().upper()
    tables = [bands["local_aliases"]["generic"]] + ([bands["local_aliases"][family]] if family else [])
    for table in tables:
        for canon, names in table.items():
            if t in (n.upper() for n in names):
                return str(canon)
    return None


def _file_token(name: str, family: str | None) -> str | None:
    for tok in re.split(r"[_.\-\s]", Path(name).stem):
        if tok and (c := canonical_for(tok, family)):
            return c
    return None


def _scale_offset(
    ds: rasterio.io.DatasetReader, idx: int, tags: dict[str, str], man: dict[str, Any]
) -> tuple[float, float, str]:
    if "scale" in man:
        return float(man["scale"]), float(man.get("offset", 0.0)), "manifest"
    s, o = ds.scales[idx - 1], ds.offsets[idx - 1]
    if (s, o) != (1.0, 0.0):
        return float(s), float(o), "gdal_metadata"
    for sk, ok in (
        ("scale_factor", "add_offset"),
        ("SCALE", "OFFSET"),
        ("REFLECTANCE_MULT", "REFLECTANCE_ADD"),
    ):
        if sk in tags:
            return float(tags[sk]), float(tags.get(ok, 0.0)), f"tag:{sk}"
    if ds.dtypes[idx - 1].startswith("float"):
        return 1.0, 0.0, "float_assumed_reflectance"
    raise LocalStageError(
        f"{ds.name}: integer DNs with no scale (no GDAL scale/offset or scale_factor tag): give scale/offset in --manifest"
    )


def _inspect(path: Path, man: dict[str, Any]) -> dict[str, Any]:
    """date, family and {canonical: (path, band index, scale, offset, nodata, how)} for one input file."""
    with rasterio.open(path) as ds:
        tags = {str(k): str(v) for k, v in ds.tags().items()}
        when = man.get("date") or detect_date(path.name, tags)
        if not when:
            raise LocalStageError(
                f"{path.name}: no acquisition date in the file name or tags: give `date` in --manifest"
            )
        fam = man.get("sensor") or sensor_family(path.name, tags)
        mapping: dict[str, int] = {}
        if "bands" in man:
            tokens = {t.upper() for t in re.split(r"[_.\-\s]", path.stem)}
            for canon, ref in man["bands"].items():
                if isinstance(ref, int) and 1 <= ref <= ds.count:
                    mapping[canon] = ref
                elif isinstance(ref, str) and ds.count == 1 and ref.upper() in tokens:
                    mapping[canon] = 1
        elif ds.count == 1:
            if c := _file_token(path.name, fam):
                mapping[c] = 1
        else:
            for i, desc in enumerate(ds.descriptions, 1):
                if desc and (c := canonical_for(desc, fam)):
                    mapping[c] = i
        if not mapping:
            raise LocalStageError(f"{path.name}: cannot tell which band is which: give `bands` in --manifest")
        nodata = man.get("nodata", ds.nodata)
        out = {}
        for canon, i in mapping.items():
            s, o, how = _scale_offset(ds, i, tags, man)
            out[canon] = (path, i, s, o, nodata, how)
        return {
            "date": str(when),
            "family": fam,
            "bands": out,
            "crs": ds.crs,
            "bounds": ds.bounds,
            "res_crs": (ds.crs, ds.width, ds.height, ds.bounds),
        }


def _native_gsd_m(ds_crs: Any, width: int, height: int, b: Any, epsg: int) -> float:
    tr, _, _ = calculate_default_transform(ds_crs, f"EPSG:{epsg}", width, height, *b)
    return float(min(abs(tr.a), abs(tr.e)))


def stage(
    inputs: list[Path],
    aoi: str,
    source: str = "organiser",
    manifest: dict[str, Any] | None = None,
    raw_root: Path | None = None,
    aoi_dir: Path | None = None,
) -> list[dict[str, Any]]:
    raw_root = raw_root or paths.raw
    aoi_dir = aoi_dir or (paths.root / "configs" / "aoi")
    man = manifest or {}
    infos = [_inspect(Path(p), man) for p in inputs]

    aoi_file = aoi_dir / f"{aoi}.yaml"
    if aoi_file.exists():
        cfg = yaml.safe_load(aoi_file.read_text(encoding="utf-8"))
        if cfg.get("optical_source", "sentinel2") != source:  # ingest reads only the AOI's optical_source
            raise LocalStageError(
                f"{aoi_file.name} has optical_source {cfg.get('optical_source', 'sentinel2')!r}; add "
                f"'optical_source: {source}' (or remove the file to have it generated) so make ingest reads these files"
            )
        bbox = tuple(cfg["bbox"])
    else:
        ll = [transform_bounds(i["crs"], 4326, *i["bounds"]) for i in infos]
        bbox = (min(b[0] for b in ll), min(b[1] for b in ll), max(b[2] for b in ll), max(b[3] for b in ll))
        aoi_dir.mkdir(parents=True, exist_ok=True)
        aoi_file.write_text(
            yaml.safe_dump(
                {
                    "name": aoi,
                    "type": "organiser",
                    "why": f"staged from local files ({source})",
                    "bbox": [round(v, 6) for v in bbox],
                    "optical_source": source,
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
    epsg = tiling.aoi_epsg(bbox)
    bounds = tiling.aoi_bounds_m(bbox, epsg)
    native = min(_native_gsd_m(*i["res_crs"], epsg) for i in infos)
    stride = T.tiling.STRIDE_M
    gsd = stride / max(
        1, round(stride / native)
    )  # divides the stride: tiles are whole pixels at any GSD (§2.9)
    width, height = round((bounds[2] - bounds[0]) / gsd), round((bounds[3] - bounds[1]) / gsd)
    transform = from_origin(bounds[0], bounds[3], gsd, gsd)

    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for i in infos:
        by_date[i["date"]].append(i)
    results = []
    for when, group in sorted(by_date.items()):
        out_dir = raw_root / source / aoi / when
        if (out_dir / "scene.json").exists():
            results.append({"date": when, "status": "exists"})
            continue
        tmp = out_dir.with_name(when + ".partial")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        band_meta: dict[str, Any] = {}
        for canon in CANON:
            srcs = [g["bands"][canon] for g in group if canon in g["bands"]]
            if not srcs:
                continue
            if len({(s[2], s[3]) for s in srcs}) > 1:
                raise LocalStageError(
                    f"{when} {canon}: mixed scale/offset across files; mosaic would be wrong"
                )
            mosaic = None
            for path, idx, _s, _o, nodata, _how in srcs:
                with rasterio.open(path) as ds:
                    nd = (
                        nodata
                        if nodata is not None
                        else (np.nan if ds.dtypes[idx - 1].startswith("float") else 0)
                    )
                    with WarpedVRT(
                        ds,
                        crs=f"EPSG:{epsg}",
                        transform=transform,
                        width=width,
                        height=height,
                        resampling=Resampling.bilinear,
                        nodata=nd,
                    ) as vrt:
                        arr = vrt.read(idx)
                if mosaic is None:
                    mosaic, fill = arr, nd
                else:
                    empty = (
                        np.isnan(mosaic) if isinstance(fill, float) and math.isnan(fill) else mosaic == fill
                    )
                    mosaic[empty] = arr[empty]
            assert mosaic is not None
            f = tmp / f"{canon.upper()}.tif"
            nd_out = None if isinstance(fill, float) and math.isnan(fill) else fill
            with rasterio.open(
                f,
                "w",
                driver="COG",
                width=width,
                height=height,
                count=1,
                dtype=mosaic.dtype,
                crs=f"EPSG:{epsg}",
                transform=transform,
                nodata=fill,
                compress="deflate",
            ) as dst:
                dst.write(mosaic, 1)
            band_meta[canon.upper()] = {
                "file": f.name,
                "sha256": sha256(f),
                "scale": srcs[0][2],
                "offset": srcs[0][3],
                "nodata": nd_out,
                "scale_source": srcs[0][5],
                "sources": [f"{s[0].name}#{s[1]}" for s in srcs],
            }
        originals = sorted({s[0] for g in group for s in g["bands"].values()}, key=lambda p: p.name)
        meta = {
            "aoi": aoi,
            "sensor": source,
            "sensor_family": group[0]["family"],
            "date": when,
            "datetime": f"{when}T00:00:00Z",
            "epsg": epsg,
            "bounds": list(bounds),
            "gsd": gsd,
            "collection": "local",
            "source_items": [p.name for p in originals],
            "source_sha256": {p.name: sha256(p) for p in originals},
            "canonical": {k.lower(): k for k in band_meta},
            "cloud_mask": "none",
            "properties": {},
            "bands": band_meta,
        }
        (tmp / "scene.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        tmp.rename(out_dir)
        results.append({"date": when, "status": "staged", "bands": sorted(band_meta), "gsd": gsd})
    return results
