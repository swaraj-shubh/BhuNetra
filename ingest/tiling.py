"""Fixed AOI tile grid in metres (DESIGN.md §2.9, §7). Resolution-agnostic: pixels = metres / GSD.

The grid origin is the AOI bbox (in its UTM zone) snapped outward to multiples of STRIDE_M, so the same place has
the same tile_id on every date and sensor, and overlapping AOIs share footprints.
"""

import math
from collections.abc import Iterator
from dataclasses import dataclass

from affine import Affine
from pyproj import Transformer
from rasterio.windows import Window, from_bounds

from configs.settings import T


@dataclass(frozen=True)
class Tile:
    tile_id: str
    row: int
    col: int
    bounds_m: tuple[float, float, float, float]  # xmin, ymin, xmax, ymax in the AOI's UTM CRS


def utm_epsg(lon: float, lat: float) -> int:
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1


def utm_zone(epsg: int) -> str:
    return f"{epsg % 100}{'N' if epsg < 32700 else 'S'}"


def aoi_epsg(bbox: tuple[float, float, float, float]) -> int:
    return utm_epsg((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def aoi_bounds_m(bbox: tuple[float, float, float, float], epsg: int) -> tuple[float, float, float, float]:
    """AOI bbox in UTM, snapped outward to STRIDE_M so an integer number of tiles covers it exactly."""
    stride, tile = T.tiling.STRIDE_M, T.tiling.TILE_M
    xmin, ymin, xmax, ymax = Transformer.from_crs(4326, epsg, always_xy=True).transform_bounds(*bbox)
    xmin, ymax = math.floor(xmin / stride) * stride, math.ceil(ymax / stride) * stride
    xmax = xmin + math.ceil(max(xmax - xmin - tile, 0) / stride) * stride + tile
    ymin = ymax - math.ceil(max(ymax - ymin - tile, 0) / stride) * stride - tile
    return xmin, ymin, xmax, ymax


def tiles(aoi: str, bounds_m: tuple[float, float, float, float], epsg: int) -> Iterator[Tile]:
    """Row 0 is the northernmost row (raster order)."""
    stride, tile = T.tiling.STRIDE_M, T.tiling.TILE_M
    xmin, ymin, xmax, ymax = bounds_m
    n_rows = round((ymax - ymin - tile) / stride) + 1
    n_cols = round((xmax - xmin - tile) / stride) + 1
    zone = utm_zone(epsg)
    for r in range(n_rows):
        for c in range(n_cols):
            x0, y1 = xmin + c * stride, ymax - r * stride
            yield Tile(f"{aoi}_{zone}_{r:04d}_{c:04d}", r, c, (x0, y1 - tile, x0 + tile, y1))


def tile_px(gsd_m: float) -> float:
    return T.tiling.TILE_M / gsd_m


def window(tile: Tile, transform: Affine) -> Window:
    """Pixel window of a tile in any raster on the AOI grid (any GSD)."""
    return from_bounds(*tile.bounds_m, transform=transform)


def footprint_lonlat(bounds_m: tuple[float, float, float, float], epsg: int) -> list[list[float]]:
    """Closed lon/lat ring (for maps and geo payloads)."""
    t = Transformer.from_crs(epsg, 4326, always_xy=True)
    x0, y0, x1, y1 = bounds_m
    ring = [t.transform(x, y) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0))]
    return [[round(x, 6), round(y, 6)] for x, y in ring]
