"""Per-tile land-cover shares per year (independent IO annual LULC v02, 10 m). Computed at ingest (§2.1).

Used to rank area-type queries ("forest", "farmland", "urban", "lake", "bare ground"): a text-image model alone is
weak on these at 10 m (M1 part 2). Linear features (river, creek, road) are never ranked by land cover: a 640 m
tile's "water" share mixes rivers with lakes, sea and salt pans. An AOI without staged LULC simply has no rows.
"""

import json

import numpy as np
import rasterio

from configs.settings import paths
from ingest import tiling
from ingest.validate import sha256

# io-lulc-annual-v02 class codes per group (0 nodata, 9 snow, 10 clouds ignored)
GROUPS = {
    "water": (1,),
    "trees": (2,),
    "flooded_vegetation": (4,),
    "crops": (5,),
    "built": (7,),
    "bare": (8,),
    "rangeland": (11,),
}


def fractions(block: np.ndarray) -> dict[str, float] | None:
    known = np.isin(block, [c for cs in GROUPS.values() for c in cs])
    if not known.any():
        return None
    v = block[known]
    return {g: round(float(np.isin(v, cs).mean()), 4) for g, cs in GROUPS.items()}


def compute(aoi: str, tiles: list[tiling.Tile]) -> dict[int, tuple[list[dict | None], dict]]:
    """year -> (fracs per tile in `tiles` order, lineage)."""
    out = {}
    for meta_file in sorted((paths.raw / "io_lulc" / aoi).glob("*/scene.json")):
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        band = meta["bands"]["LULC"]
        tif = meta_file.parent / band["file"]
        if sha256(tif) != band["sha256"]:
            raise ValueError(f"sha256 mismatch: {tif}")
        with rasterio.open(tif) as ds:
            arr, tr = ds.read(1), ds.transform
        fr = []
        for t in tiles:
            w = tiling.window(t, tr)
            r0, c0 = max(round(w.row_off), 0), max(round(w.col_off), 0)
            fr.append(fractions(arr[r0 : r0 + round(w.height), c0 : c0 + round(w.width)]))
        lin = {"file": str(tif.relative_to(paths.root)), "sha256": band["sha256"], "groups": GROUPS}
        out[int(meta_file.parent.name)] = (fr, lin)
    return out
