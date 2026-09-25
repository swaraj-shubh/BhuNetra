import numpy as np
import pytest
from pydantic import ValidationError
from rasterio.transform import from_origin

from configs.settings import T, Thresholds
from ingest import masks, standardise, tiling
from ingest.stac import DB_NODATA, linear_to_db_int16

NAVI = (72.98, 18.93, 73.17, 19.11)


def test_reflectance_applies_offset_and_nodata():
    dn = np.array([[0, 1000, 2000]], dtype=np.uint16)
    r = standardise.to_reflectance(dn, 0.0001, -0.1)
    assert np.isnan(r[0, 0])
    assert np.allclose(r[0, 1:], [0.0, 0.1])  # post-2022 L2A: DN 1000 == 0 reflectance


def test_s2_mask_rejects_scl_cloud_and_cld_prob_with_dilation():
    scl = np.full((30, 30), 4, dtype=np.uint8)  # vegetation
    scl[5, 5] = 9  # SCL high-probability cloud
    cld = np.zeros((30, 30), dtype=np.uint8)
    cld[20, 20] = int(T.quality.CLOUD_PROB * 100)  # exactly at threshold -> cloudy
    valid = masks.s2_valid_mask(scl, cld)
    d = T.quality.CLOUD_DILATE_PX
    for y, x in ((5, 5), (20, 20)):
        assert not valid[y - d : y + d + 1, x - d : x + d + 1].any()
    assert valid.sum() == 900 - 2 * (2 * d + 1) ** 2


def test_tile_size_in_metres_at_any_gsd():
    assert tiling.tile_px(10) == 64 and tiling.tile_px(1) == 640
    tile = next(tiling.tiles("t", (0.0, 0.0, 6400.0, 6400.0), 32643))
    for gsd in (10.0, 1.0, 30.0):
        w = tiling.window(tile, from_origin(0.0, 6400.0, gsd, gsd))
        assert (w.width, w.height) == pytest.approx((T.tiling.TILE_M / gsd,) * 2)


def test_tile_ids_stable_and_footprints_shared_across_aois():
    epsg = tiling.aoi_epsg(NAVI)
    a = list(tiling.tiles("navimumbai", tiling.aoi_bounds_m(NAVI, epsg), epsg))
    again = list(tiling.tiles("navimumbai", tiling.aoi_bounds_m(NAVI, epsg), epsg))
    assert a == again  # deterministic: same place, same id on every run/date
    assert a[0].tile_id == "navimumbai_43N_0000_0000"
    other = {
        t.bounds_m for t in tiling.tiles("x", tiling.aoi_bounds_m((73.05, 19.0, 73.2, 19.15), epsg), epsg)
    }
    shared = {t.bounds_m for t in a} & other
    assert shared, "overlapping AOIs must share tile footprints"
    x0, y0, x1, y1 = a[0].bounds_m
    assert (x1 - x0, y1 - y0) == (T.tiling.TILE_M, T.tiling.TILE_M)
    assert x0 % T.tiling.STRIDE_M == 0 and y1 % T.tiling.STRIDE_M == 0


def test_aoi_bounds_cover_bbox_with_whole_tiles():
    epsg = tiling.aoi_epsg(NAVI)
    xmin, ymin, xmax, ymax = tiling.aoi_bounds_m(NAVI, epsg)
    for span in (xmax - xmin, ymax - ymin):
        assert (span - T.tiling.TILE_M) % T.tiling.STRIDE_M == 0


def test_sar_db_roundtrip_and_nodata():
    lin = np.array([[0.1, 1.0, 0.0, -32768.0]], dtype=np.float32)
    db = linear_to_db_int16(lin)
    assert db.tolist() == [[-1000, 0, DB_NODATA, DB_NODATA]]  # 0.1 linear == -10 dB
    back = standardise.db_to_linear(db)
    assert np.allclose(back[0, :2], lin[0, :2], rtol=1e-3) and np.isnan(back[0, 2:]).all()


def test_thresholds_reject_unknown_keys():
    bad = T.model_dump()
    bad["quality"]["Q_MINN"] = 0.5  # typo must fail loudly, not be silently ignored
    with pytest.raises(ValidationError):
        Thresholds(**bad)
