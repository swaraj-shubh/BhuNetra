"""Read one staged acquisition folder (data/raw/<source>/<aoi>/<key>/, DESIGN.md §9) into georeferenced arrays.

Read-only on data/raw (§2.12). Every band file is checked against its sha256 in scene.json (§2.7).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

from configs.settings import bands
from ingest import masks
from ingest.standardise import to_reflectance
from ingest.validate import sha256


class SceneRejected(Exception):
    """A bad scene: stored as rejected with this reason, never crashes the batch (§10)."""


@dataclass
class Observation:
    scene_id: str
    source: str
    aoi: str
    date: str
    folder: Path
    meta: dict  # scene.json
    epsg: int
    transform: Affine
    refl: dict[str, np.ndarray]  # canonical band -> float32 reflectance, NaN = nodata
    valid: np.ndarray  # bool: usable pixel (all bands present, not invalid SCL, not dilated cloud)
    cloud: np.ndarray  # bool: dilated cloud/shadow (for cloud_near)

    @property
    def sun_elev(self) -> float | None:
        return self.meta.get("properties", {}).get("view:sun_elevation")


def read_meta(folder: Path) -> dict:
    return json.loads((folder / "scene.json").read_text(encoding="utf-8"))


def fingerprint(folder: Path) -> str:
    """Idempotency key: scene.json carries every band's sha256, so its hash identifies the whole acquisition."""
    return sha256(folder / "scene.json")


def _read(folder: Path, band: dict, expect_transform: Affine | None) -> tuple[np.ndarray, Affine, int]:
    path = folder / band["file"]
    if sha256(path) != band["sha256"]:
        raise SceneRejected(f"sha256 mismatch: {path.name}")
    with rasterio.open(path) as d:
        if expect_transform is not None and d.transform != expect_transform:
            raise SceneRejected(f"{path.name} is not on the scene grid")
        return d.read(1), d.transform, d.crs.to_epsg()


def load_s2(folder: Path) -> Observation:
    meta = read_meta(folder)
    files = meta["bands"]
    scl, transform, epsg = _read(folder, files[bands["sentinel2"]["mask_bands"]["scl"]], None)
    cld_name = bands["sentinel2"]["mask_bands"]["cloud_prob"]
    cld = _read(folder, files[cld_name], transform)[0] if cld_name in files else None

    refl, present = {}, np.ones(scl.shape, bool)
    for canon, name in bands["sentinel2"]["bands"].items():
        b = files[name]
        dn = _read(folder, b, transform)[0]
        refl[canon] = to_reflectance(dn, b["scale"], b["offset"], b.get("nodata"))  # effective offset (§9)
        present &= ~np.isnan(refl[canon])

    return Observation(
        scene_id="+".join(meta["source_items"]),
        source="sentinel2",
        aoi=meta["aoi"],
        date=meta["date"],
        folder=folder,
        meta=meta,
        epsg=epsg,
        transform=transform,
        refl=refl,
        valid=present & masks.s2_valid_mask(scl, cld),
        cloud=masks.s2_cloud_mask(scl, cld),
    )


def load_generic(folder: Path) -> Observation:
    """Local / organiser scene (ingest/local.py): bands from scene.json["canonical"], no cloud mask unless provided.

    Missing canonical bands are simply absent from `refl` (e.g. RGB-only imagery: search works, indices do not).
    """
    meta = read_meta(folder)
    files = meta["bands"]
    refl: dict[str, np.ndarray] = {}
    transform = epsg = present = None
    for canon, name in meta["canonical"].items():
        b = files[name]
        dn, transform, epsg = _read(folder, b, transform)
        refl[canon] = to_reflectance(dn, b["scale"], b["offset"], b.get("nodata"))
        ok = ~np.isnan(refl[canon])
        present = ok if present is None else present & ok
    if present is None:
        raise SceneRejected("scene.json lists no bands")
    return Observation(
        scene_id="+".join(meta["source_items"]),
        source=meta["sensor"],
        aoi=meta["aoi"],
        date=meta["date"],
        folder=folder,
        meta=meta,
        epsg=epsg,
        transform=transform,
        refl=refl,
        valid=present,
        cloud=np.zeros_like(present),  # cloud_mask: none -> persistence is the cloud defence (§2.5)
    )


def load(folder: Path) -> Observation:
    """Any staged scene: STAC-staged Sentinel-2 (SCL/CLD masks) or local/organiser files (generic)."""
    return load_generic(folder) if "canonical" in read_meta(folder) else load_s2(folder)
