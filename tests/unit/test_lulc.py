import numpy as np

from worker.lulc_evidence import class_fracs, verdict


def test_class_fracs_ignore_nodata_and_clouds():
    lulc = np.array([[7, 7, 2, 0], [10, 1, 8, 5]], np.uint8)
    f = class_fracs(lulc, np.ones_like(lulc, bool))
    assert f == {"water": 0.1667, "built": 0.3333, "bare": 0.1667, "vegetation": 0.3333}
    assert class_fracs(np.zeros((2, 2), np.uint8), np.ones((2, 2), bool)) is None


def test_verdict_types_and_no_claims():
    veg = {"water": 0.0, "built": 0.0, "bare": 0.0, "vegetation": 1.0}
    built = {"water": 0.0, "built": 0.6, "bare": 0.1, "vegetation": 0.3}
    bare = {"water": 0.0, "built": 0.0, "bare": 0.7, "vegetation": 0.3}
    assert verdict("construction", veg, built) == ("construction", True)
    assert verdict("construction", veg, bare) == ("clearance", False)
    assert verdict("water", veg, veg) == ("stable", False)
    assert verdict("road", veg, built) == ("construction", None)  # roads: no claim
    assert verdict("clearance", None, bare) == (None, None)  # year not staged
