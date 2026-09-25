import numpy as np
import pytest
from rasterio.transform import from_origin

from configs.settings import T
from features import composites
from ingest import tiling


@pytest.mark.parametrize(
    ("date", "season"),
    [
        ("2023-12-15", "2024-DJF"),
        ("2024-01-05", "2024-DJF"),
        ("2024-02-29", "2024-DJF"),
        ("2024-03-01", "2024-MAM"),
        ("2024-09-30", "2024-JJAS"),
        ("2024-11-30", "2024-ON"),
    ],
)
def test_season_of(date, season):
    assert composites.season_of(date) == season


def test_season_window_matches_season_of():
    start, end = composites.season_window("2024-DJF")
    assert (start, end) == ("2023-12-01", "2024-02-29")
    assert composites.season_window("2023-ON") == ("2023-10-01", "2023-11-30")


def test_median_ignores_invalid_pixels_and_counts_obs():
    a = np.array([[0.1, 0.2]], np.float32)
    b = np.array([[0.3, 0.9]], np.float32)
    c = np.array([[0.5, 0.4]], np.float32)
    usable = [np.array([[True, True]]), np.array([[True, False]]), np.array([[True, True]])]
    comp, n = composites.median_composite([{"red": a}, {"red": b}, {"red": c}], usable)
    assert comp["red"][0].tolist() == pytest.approx([0.3, 0.3])  # median(0.1,0.3,0.5); median(0.2,0.4)
    assert n.tolist() == [[3, 2]]


def test_median_all_invalid_is_nan():
    comp, n = composites.median_composite([{"red": np.ones((1, 1), np.float32)}], [np.zeros((1, 1), bool)])
    assert np.isnan(comp["red"][0, 0]) and n[0, 0] == 0


def test_q_ok_mask_paints_only_good_tiles():
    gsd, px = 10.0, int(T.tiling.TILE_M / 10.0)
    stride = int(T.tiling.STRIDE_M / 10.0)
    h = w = px + stride  # 2x2 tiles
    transform = from_origin(320_000.0, 2_000_000.0, gsd, gsd)
    tl = list(tiling.tiles("t", (320_000.0, 2_000_000.0 - h * gsd, 320_000.0 + w * gsd, 2_000_000.0), 32643))
    q = np.array([T.quality.Q_MIN if (t.row, t.col) == (0, 0) else 0.0 for t in tl])
    ok = composites.q_ok_mask(q, tl, transform, (h, w))
    assert ok[:px, :px].all() and not ok[px:, :].any() and not ok[:, px:].any()
