"""Frozen spatial-block split (M5; DESIGN.md §2.10, §18 C6). Deterministic: salted sha256, no RNG.

Every AOI is cut into BLOCK_M blocks from its grid origin; per AOI the round(HELDOUT_FRAC * n) blocks (min 1) with the
lowest hash are held out. A tile belongs to the block containing its centroid. Held-out labels are only read in
eval/ for the final report (§9).

    python eval/split.py        # writes eval/splits/m5_blocks.json (commit it; never re-draw after results)
"""

import hashlib
import json
import math

from configs.settings import aois, eval_cfg, paths
from ingest import tiling

CFG = eval_cfg["split"]
OUT = paths.root / "eval" / "splits" / "m5_blocks.json"


def block_of(
    centroid_m: tuple[float, float], aoi_bounds_m: tuple[float, float, float, float]
) -> tuple[int, int]:
    xmin, _, _, ymax = aoi_bounds_m
    return int((centroid_m[0] - xmin) // CFG["BLOCK_M"]), int((ymax - centroid_m[1]) // CFG["BLOCK_M"])


def block_id(aoi: str, b: tuple[int, int]) -> str:
    return f"{aoi}_b{b[0]}_{b[1]}"


def _h(bid: str) -> int:
    return int(hashlib.sha256(f"{CFG['SALT']}|{bid}".encode()).hexdigest()[:12], 16)


def heldout_blocks(aoi: str, bounds_m: tuple[float, float, float, float]) -> list[str]:
    nx = math.ceil((bounds_m[2] - bounds_m[0]) / CFG["BLOCK_M"])
    ny = math.ceil((bounds_m[3] - bounds_m[1]) / CFG["BLOCK_M"])
    ids = sorted((block_id(aoi, (i, j)) for i in range(nx) for j in range(ny)), key=_h)
    return sorted(ids[: max(1, round(CFG["HELDOUT_FRAC"] * len(ids)))])


def build() -> dict:
    out = {"config": CFG, "aois": {}}
    for name, a in sorted(aois.items()):
        epsg = tiling.aoi_epsg(a.bbox)
        bounds = tiling.aoi_bounds_m(a.bbox, epsg)
        tiles = list(tiling.tiles(name, bounds, epsg))
        held = set(heldout_blocks(name, bounds))
        blocks = {
            t.tile_id: block_id(
                name,
                block_of(((t.bounds_m[0] + t.bounds_m[2]) / 2, (t.bounds_m[1] + t.bounds_m[3]) / 2), bounds),
            )
            for t in tiles
        }
        out["aois"][name] = {
            "heldout_blocks": sorted(held),
            "n_tiles": len(tiles),
            "n_heldout_tiles": sum(b in held for b in blocks.values()),
        }
    return out


def split_of(tile_id: str, frozen: dict | None = None) -> str:
    """'heldout' or 'calibration' for a tile, from the frozen split file."""
    frozen = frozen or json.loads(OUT.read_text(encoding="utf-8"))
    aoi = tile_id.split("_")[0]
    a = aois[aoi]
    epsg = tiling.aoi_epsg(a.bbox)
    bounds = tiling.aoi_bounds_m(a.bbox, epsg)
    _, _, r, c = tile_id.rsplit("_", 3)
    x0 = bounds[0] + int(c) * tiling.T.tiling.STRIDE_M
    y1 = bounds[3] - int(r) * tiling.T.tiling.STRIDE_M
    half = tiling.T.tiling.TILE_M / 2
    bid = block_id(aoi, block_of((x0 + half, y1 - half), bounds))
    return "heldout" if bid in frozen["aois"][aoi]["heldout_blocks"] else "calibration"


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        raise SystemExit(
            f"{OUT} exists: the split is frozen. Delete it only if no results were seen with it."
        )
    data = build()
    OUT.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(
        json.dumps(
            {
                k: {kk: v[kk] for kk in ("heldout_blocks", "n_heldout_tiles", "n_tiles")}
                for k, v in data["aois"].items()
            },
            indent=1,
        )
    )
