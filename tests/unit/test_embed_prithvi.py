import numpy as np
import pytest

from features.embed_prithvi import normalise, pool_tiles, windows


def test_windows_cover_the_raster_with_224px_blocks():
    ws = windows(300, 500, 224)
    assert ws[0] == (0, 0) and (224, 448) in ws  # padded grid
    assert len(ws) == 2 * 3


def test_pool_tiles_averages_the_patch_tokens_inside_each_tile():
    fmap = np.zeros((8, 8, 2), np.float32)  # 8x8 patches of 16 px = 128x128 px
    fmap[0:4, 0:4] = [1, 0]  # tile at px (0, 0), 64 px = 4 patches
    fmap[2:6, 4:8] = [0, 1]  # tile at px (32, 64)
    out = pool_tiles(fmap, [(0, 0), (32, 64)], tile_px=64, patch_px=16)
    assert np.allclose(out[0], [1, 0])
    assert np.allclose(out[1], [0, 1])
    assert np.allclose(np.linalg.norm(out, axis=1), 1)  # L2-normalised


def test_pool_tiles_rejects_misaligned_tiles():
    with pytest.raises(ValueError):
        pool_tiles(np.zeros((8, 8, 2), np.float32), [(8, 0)], tile_px=64, patch_px=16)


def test_normalise_uses_model_card_stats_and_zeroes_nodata():
    x = np.full((6, 2, 2), 0.1, np.float32)
    x[:, 0, 0] = np.nan
    n = normalise(x, mean=[1000.0] * 6, std=[1000.0] * 6)
    assert np.allclose(n[:, 1, 1], 0.0)  # 0.1 reflectance = 1000 DN = the mean
    assert np.allclose(n[:, 0, 0], 0.0)  # nodata -> mean (neutral)
