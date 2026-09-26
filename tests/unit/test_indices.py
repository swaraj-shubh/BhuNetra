import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from features import indices


def refl(**v):
    return {b: np.array([[v.get(b, 0.1)]], np.float32) for b in ("blue", "green", "red", "nir", "swir1")}


def test_formulas_match_design_doc():
    r = refl(blue=0.05, green=0.08, red=0.06, nir=0.40, swir1=0.20)
    ix = indices.compute(r)
    assert ix["ndvi"][0, 0] == pytest.approx((0.40 - 0.06) / (0.40 + 0.06))
    assert ix["mndwi"][0, 0] == pytest.approx((0.08 - 0.20) / (0.08 + 0.20))
    assert ix["ndbi"][0, 0] == pytest.approx((0.20 - 0.40) / (0.20 + 0.40))
    assert ix["bsi"][0, 0] == pytest.approx(((0.20 + 0.06) - (0.40 + 0.05)) / ((0.20 + 0.06) + (0.40 + 0.05)))


def test_nan_and_zero_denominator_give_nan():
    r = refl(nir=np.nan)
    assert np.isnan(indices.compute(r)["ndvi"][0, 0])
    z = refl(nir=0.0, red=0.0)
    assert np.isnan(indices.compute(z)["ndvi"][0, 0])


def test_write_read_roundtrip_masks_unusable(tmp_path):
    h = w = 4
    r = {
        b: np.full((h, w), x, np.float32)
        for b, x in (("blue", 0.05), ("green", 0.08), ("red", 0.06), ("nir", 0.4), ("swir1", 0.2))
    }
    usable = np.ones((h, w), bool)
    usable[0, 0] = False
    f = tmp_path / "a.tif"
    indices.write(f, indices.compute(r), usable, from_origin(0, 40, 10, 10), 32643, {"scene_id": "X"})
    with rasterio.open(f) as d:
        assert list(d.descriptions) == list(indices.NAMES)
    out = indices.read(f)
    assert np.isnan(out["ndvi"][0, 0])
    assert out["ndvi"][1, 1] == pytest.approx((0.4 - 0.06) / 0.46, abs=1e-4)  # int16 x 1e4 storage
    assert (tmp_path / "a.json").exists()


def test_out_of_range_from_negative_reflectance_is_nan():
    # L2A minus the BOA offset can go slightly negative: nir + red ~ 0 makes the ratio explode
    r = refl(nir=0.011, red=-0.01)
    assert np.isnan(indices.compute(r)["ndvi"][0, 0])
