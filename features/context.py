"""Per-tile spatial context for "near a river / road" queries (M4). Computed at ingest, never per query (§2.1).

water: JRC Global Surface Water occurrence >= WATER_OCCURRENCE_MIN %, speckle removed (< WATER_MIN_PX pixels).
       JRC cannot tell rivers from lakes, creeks or sea: every target word in configs/query.yaml `water` maps here.
road:  latest staged OSM snapshot, classes in ROAD_CLASSES.
Distance = metres from the tile footprint to the nearest feature (0 when they touch); inf when the AOI has none.
"""

import json
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import shapes, sieve
from shapely import STRtree
from shapely.geometry import LineString, box, shape
from shapely.ops import transform

from configs.settings import T, paths
from ingest import tiling
from ingest.validate import sha256


def water_geoms(aoi: str) -> tuple[list, dict]:
    folder = paths.raw / "jrc_gsw" / aoi / "static"
    if not (
        folder / "scene.json"
    ).exists():  # e.g. an organiser AOI without staged JRC: "near water" matches nothing
        return [], {"file": None}
    meta = json.loads((folder / "scene.json").read_text(encoding="utf-8"))
    band = meta["bands"]["OCCURRENCE"]
    tif = folder / band["file"]
    if sha256(tif) != band["sha256"]:
        raise ValueError(f"sha256 mismatch: {tif}")
    with rasterio.open(tif) as ds:
        occ = ds.read(1)
        tf = ds.transform
    # 0 = never water (the staged `nodata: 0` is a catalog artefact, not missing data); > 100 = invalid
    water = ((occ >= T.context.WATER_OCCURRENCE_MIN) & (occ <= 100)).astype(np.uint8)
    water = sieve(water, size=T.context.WATER_MIN_PX)
    geoms = [shape(g) for g, v in shapes(water, mask=water == 1, transform=tf) if v == 1]
    return geoms, {"file": str(tif.relative_to(paths.root)), "sha256": band["sha256"]}


def road_geoms(aoi: str, epsg: int) -> tuple[list, dict]:
    snaps = sorted((paths.raw / "osm" / aoi).glob("roads_*.geojson"))
    if not snaps:
        return [], {"file": None}
    latest: Path = snaps[-1]
    fc = json.loads(latest.read_text(encoding="utf-8"))
    to_utm = Transformer.from_crs(4326, epsg, always_xy=True).transform
    keep = set(T.context.ROAD_CLASSES)
    geoms = [
        transform(to_utm, LineString(f["geometry"]["coordinates"]))
        for f in fc["features"]
        if f["properties"].get("highway") in keep and f["geometry"]["type"] == "LineString"
    ]
    return geoms, {"file": str(latest.relative_to(paths.root)), "sha256": sha256(latest)}


def distances(tiles: list[tiling.Tile], geoms: list) -> np.ndarray:
    """Metres from each tile footprint to its nearest geometry (vectorised STRtree nearest)."""
    if not geoms:
        return np.full(len(tiles), np.inf)
    polys = [box(*t.bounds_m) for t in tiles]
    idx, dist = STRtree(geoms).query_nearest(polys, return_distance=True, all_matches=False)
    out = np.full(len(tiles), np.inf)
    out[idx[0]] = dist
    return out


def compute(aoi: str, tiles: list[tiling.Tile], epsg: int) -> dict[str, tuple[np.ndarray, dict]]:
    params = T.context.model_dump()
    water, w_in = water_geoms(aoi)
    roads, r_in = road_geoms(aoi, epsg)
    return {
        "water": (distances(tiles, water), {"inputs": w_in, "params": params, "n_features": len(water)}),
        "road": (distances(tiles, roads), {"inputs": r_in, "params": params, "n_features": len(roads)}),
    }
