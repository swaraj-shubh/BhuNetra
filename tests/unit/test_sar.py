import numpy as np
import pytest

from change.sar import agrees, log_ratio_db, stack_median_db


def test_stack_median_is_computed_in_linear_power_and_ignores_nodata():
    nod = -32768
    # two dates at -10 dB and -20 dB -> linear median of (0.1, 0.01) = 0.055 -> -12.6 dB (not the dB mean -15)
    a = np.full((2, 2), -1000, np.int16)
    b = np.full((2, 2), -2000, np.int16)
    b[0, 0] = nod
    mask = np.ones((2, 2), bool)
    v = stack_median_db([a, b], mask, nodata=nod)
    assert v == pytest.approx(10 * np.log10(np.median([0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01])), abs=1e-6)


def test_log_ratio_and_empty_inputs():
    assert log_ratio_db(-12.0, -8.5) == pytest.approx(3.5)
    assert log_ratio_db(None, -8.5) is None
    assert stack_median_db([], np.ones((2, 2), bool), -32768) is None


@pytest.mark.parametrize(
    ("change_type", "dvv", "dvh", "expected"),
    [
        ("construction", 2.5, 0.5, True),  # double-bounce: VV up
        ("construction", 0.2, 0.1, False),
        ("clearance", 0.0, -2.0, True),  # less volume scattering: VH down
        ("water", -4.0, -3.0, True),  # smooth open water: VV down (appearance)
        ("water", 4.0, 3.0, True),  # water disappearance: VV up
        ("road", 1.0, 0.0, None),  # no defined SAR signature -> no claim
        ("other", -3.0, -3.0, None),
    ],
)
def test_agreement_rules(change_type, dvv, dvh, expected):
    assert agrees(change_type, dvv, dvh) is expected
    assert agrees(change_type, None, dvh) is None  # missing S1 never counts as evidence
