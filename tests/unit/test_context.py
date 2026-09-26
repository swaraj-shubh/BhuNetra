import numpy as np
from shapely.geometry import LineString, box

from features.context import distances
from ingest.tiling import Tile


def _tile(x0: float) -> Tile:
    return Tile(f"t{x0:.0f}", 0, 0, (x0, 0.0, x0 + 640.0, 640.0))


def test_distances_touching_far_and_none():
    tiles = [_tile(0), _tile(2000)]
    river = [LineString([(700, -100), (700, 800)])]  # 60 m east of tile 0, 1300 m west of tile 1
    lake = [box(-100, 0, 0, 640)]  # touches tile 0
    assert np.allclose(distances(tiles, river), [60.0, 1300.0])
    assert distances(tiles, lake)[0] == 0.0
    assert np.isinf(distances(tiles, [])).all()  # no features in the AOI -> "near X" cannot match
