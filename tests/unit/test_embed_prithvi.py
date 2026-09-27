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


def test_pool_tiles_misaligned_tile_uses_every_overlapped_patch():
    # ~30 m organiser imagery: 22 px tiles on 16 px patches
    fmap = np.zeros((4, 4, 2), np.float32)
    fmap[0:3, 1:3] = [0, 1]  # patches overlapped by a 22 px tile at px (11, 22): rows 0..2, cols 1..2
    out = pool_tiles(fmap, [(11, 22)], tile_px=22, patch_px=16)
    assert np.allclose(out[0], [0, 1])


def test_bands_used_falls_back_to_broad_nir():
    from features.embed_prithvi import bands_used

    assert bands_used(["blue", "green", "red", "nir", "swir1", "swir2"])[3] == "nir"
    assert bands_used(["blue", "green", "red", "nir", "nir_narrow", "swir1", "swir2"])[3] == "nir_narrow"
    with pytest.raises(ValueError):
        bands_used(["blue", "green", "red", "nir"])


def test_normalise_uses_model_card_stats_and_zeroes_nodata():
    x = np.full((6, 2, 2), 0.1, np.float32)
    x[:, 0, 0] = np.nan
    n = normalise(x, mean=[1000.0] * 6, std=[1000.0] * 6)
    assert np.allclose(n[:, 1, 1], 0.0)  # 0.1 reflectance = 1000 DN = the mean
    assert np.allclose(n[:, 0, 0], 0.0)  # nodata -> mean (neutral)
