import numpy as np
import pytest

from query.rocchio import rocchio


def _n(v):
    return v / np.linalg.norm(v)


def test_confirmed_pulls_toward_and_rejected_pushes_away():
    q = _n(np.array([1.0, 0.0, 0.0]))
    good = _n(np.array([0.0, 1.0, 0.0]))
    bad = _n(np.array([0.0, 0.0, 1.0]))
    q2 = rocchio(q, [good], [bad])
    assert np.linalg.norm(q2) == pytest.approx(1.0)
    assert q2 @ good > q @ good  # pulled toward confirmed
    assert q2 @ bad < q @ bad  # pushed away from rejected


def test_no_feedback_is_identity_and_weights_match_design_doc():
    q = _n(np.array([0.3, 0.4, 0.5]))
    assert np.allclose(rocchio(q, [], []), q)
    good, bad = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    expected = _n(1.0 * q + 0.75 * good - 0.15 * bad)
    assert np.allclose(rocchio(q, [good], [bad]), expected)
