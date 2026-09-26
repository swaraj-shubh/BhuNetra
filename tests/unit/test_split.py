import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
import split


def test_split_is_deterministic_and_holds_out_at_least_one_block_per_aoi():
    a, b = split.build(), split.build()
    assert a == b
    for v in a["aois"].values():
        assert len(v["heldout_blocks"]) >= 1
        assert 0 < v["n_heldout_tiles"] < v["n_tiles"]


def test_split_of_matches_block_assignment():
    data = split.build()
    for aoi, v in data["aois"].items():
        tiles = split.tiling.tiles(aoi, *reversed(_grid(aoi)))
        labels = [split.split_of(t.tile_id, data) for t in tiles]
        assert labels.count("heldout") == v["n_heldout_tiles"]


def _grid(aoi):
    a = split.aois[aoi]
    epsg = split.tiling.aoi_epsg(a.bbox)
    return epsg, split.tiling.aoi_bounds_m(a.bbox, epsg)
