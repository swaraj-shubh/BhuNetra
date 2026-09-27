"""Landsat evidence (M9): QA_PIXEL mask, year windows, footprint medians, type agreement."""

import numpy as np

from configs.settings import T
from ingest import masks
from worker.landsat_evidence import side_median, verdict, window


def test_qa_pixel_mask_bits():
    clear, fill, cloud, shadow, snow, water_bit = 0, 1, 1 << 3, 1 << 4, 1 << 5, 1 << 7
    qa = np.array([clear, fill, cloud, shadow, snow, water_bit], np.uint16)
    assert masks.landsat_valid_mask(qa).tolist() == [True, False, False, False, False, True]
    assert masks.landsat_cloud_mask(qa).tolist() == [False, False, True, True, False, False]


def test_window_takes_the_year_on_each_side():
    dates = ["2020-01-01", "2020-06-01", "2021-01-01", "2021-02-01", "2021-12-31", "2022-02-01"]
    before, after = window(dates, "2021-01-01", "2021-02-01")
    assert before == [1, 2] and after == [3, 4]
    assert window(dates, None, "2021-02-01")[0] == []


def test_side_median_needs_min_obs_and_valid_fraction():
    n = T.landsat.MIN_OBS
    vals = np.tile(np.arange(n + 1, dtype=np.float32)[:, None], (1, 4))
    valid = np.ones_like(vals, bool)
    assert side_median(vals, valid, list(range(n))) == float(np.median(np.arange(n)))
    assert side_median(vals, valid, list(range(n - 1))) is None
    valid[0] = False  # one date fully cloudy: drops below MIN_OBS
    assert side_median(vals, valid, list(range(n))) is None


def test_verdict_agreement_and_no_claim():
    water, shape = {"pre": 0.0, "post": 0.0}, {"elongation": 1.5, "width_px": 10}
    cleared = {"ndvi": -0.4, "mndwi": 0.0, "ndbi": 0.02, "bsi": 0.15}
    assert verdict("clearance", cleared, water, shape) == ("clearance", True)
    assert verdict("construction", cleared, water, shape) == ("clearance", False)
    assert verdict("other", cleared, water, shape)[1] is None
    assert verdict("clearance", cleared | {"ndvi": None}, water, shape) == (None, None)
