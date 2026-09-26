from datetime import date, timedelta

import numpy as np
from rasterio.transform import from_origin

from change import events
from change.cascade import BANDS
from configs.settings import T

H = W = 40
DATES = [(date(2019, 1, 5) + timedelta(days=30 * i)).isoformat() for i in range(72)]
TR = from_origin(320_000.0, 2_500_000.0, 10.0, 10.0)


def empty():
    p = {b: np.full((H, W), np.nan, np.float32) for b in BANDS}
    p["water_pre"][:] = 0
    p["water_post"][:] = 0
    return p


def paint(p, rows, cols, idx, d_ndvi=-0.3, d_ndbi=0.15, d_bsi=0.0):
    p["break_idx"][rows, cols] = idx
    p["last_normal_idx"][rows, cols] = idx - 1
    p["d_ndvi"][rows, cols] = d_ndvi
    p["d_ndbi"][rows, cols] = d_ndbi
    p["d_bsi"][rows, cols] = d_bsi
    p["d_mndwi"][rows, cols] = 0.0


def test_block_becomes_one_construction_event_and_specks_are_dropped():
    p = empty()
    paint(p, slice(5, 15), slice(5, 15), 40)  # 100 px construction site
    paint(p, slice(30, 32), slice(30, 31), 40)  # 2 px speck < MIN_EVENT_PX
    ev = events.extract(p, DATES, TR, 32643)
    assert len(ev) == 1
    e = ev[0]
    assert (e.change_type, e.direction) == ("construction", "appearance")
    assert e.break_idx == 40 and e.last_normal_idx == 39
    assert e.evidence["shape"]["n_px"] == 100 and e.evidence["shape"]["area_m2"] == 10_000
    assert e.evidence["coherence"] == 1.0 and not e.evidence["calibrated"]
    assert 0 < e.confidence <= 1
    assert e.geom_wkt.startswith("MULTIPOLYGON")


def test_same_place_different_seasons_are_separate_events():
    p = empty()
    paint(p, slice(5, 15), slice(5, 15), 40)
    paint(p, slice(5, 15), slice(15, 25), 55)  # adjacent, broke ~15 months later (different season)
    ev = events.extract(p, DATES, TR, 32643)
    assert sorted(e.break_idx for e in ev) == [40, 55]


def test_min_event_px_boundary():
    n = T.events.MIN_EVENT_PX
    p = empty()
    paint(p, slice(0, 1), slice(0, n), 40)  # exactly MIN_EVENT_PX in a row: kept (and elongated -> road)
    ev = events.extract(p, DATES, TR, 32643)
    assert len(ev) == 1 and ev[0].change_type == "road"


def test_timelines_medians_per_date():
    fp = np.flatnonzero(np.pad(np.ones((2, 2), bool), 1))  # 2x2 block inside a 4x4 raster
    d1 = {"ndvi": np.full((4, 4), 0.5, np.float32)}
    d2 = {"ndvi": np.full((4, 4), np.nan, np.float32)}  # cloudy date: no value
    assert events.timelines([fp], [d1, d2]) == [{"ndvi": [0.5, None]}]
