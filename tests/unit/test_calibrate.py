import numpy as np
import pytest

from change.calibrate import Calibration, fit_isotonic


def test_isotonic_is_monotone_and_matches_pav_on_a_known_case():
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    y = np.array([0, 1, 0, 0, 1, 1])
    cal = fit_isotonic(x, y)
    p = cal.predict(x)
    assert np.all(np.diff(p) >= 0)
    # PAV pools (1, 0, 0) -> 1/3 for x = 0.2..0.4
    assert p.tolist() == pytest.approx([0.0, 1 / 3, 1 / 3, 1 / 3, 1.0, 1.0])


def test_predict_interpolates_and_clamps():
    cal = fit_isotonic(np.array([0.0, 1.0]), np.array([0, 1]))
    assert cal.predict(np.array([-1.0, 0.25, 2.0])).tolist() == pytest.approx([0.0, 0.25, 1.0])


def test_uninformative_score_calibrates_to_the_base_rate():
    x = np.full(10, 0.9)  # a saturated raw score carries no ranking information
    y = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0])
    assert Calibration.from_json(fit_isotonic(x, y).to_json()).predict(np.array([0.9]))[0] == pytest.approx(
        0.3
    )


def test_fit_rejects_empty_and_mismatched_input():
    with pytest.raises(ValueError):
        fit_isotonic(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        fit_isotonic(np.array([0.1, 0.2]), np.array([1]))


def test_weak_label_rules():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
    from weak_labels import label

    z = {"built": 0.0, "trees": 0.0, "water": 0.0, "flooded_vegetation": 0.0}
    assert label("construction", z | {"built": 0.3}, 0.3, 0) == "tp"
    assert label("construction", z | {"trees": -0.3}, 0.3, 0) == "fp_wrongtype"  # real change, but clearance
    assert label("construction", z, 0.02, 0) == "fp_nochange"
    assert label("construction", z | {"built": 0.1}, 0.1, 0) == "ambiguous"
    assert label("clearance", z | {"trees": -0.25}, 0.25, 0) == "tp"
    assert label("water", z | {"water": -0.1, "flooded_vegetation": -0.15}, 0.25, 0) == "tp"
    assert label("road", z, 0.02, 250) == "tp"  # stable land cover but a new road
    assert label("other", z | {"built": 0.05}, 0.22, 0) == "tp"
