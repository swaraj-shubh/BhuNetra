"""DESIGN.md §11: harmonic + breaks on synthetic seasonal series."""

from datetime import date, timedelta

import numpy as np
import pytest

from change import breaks, harmonic
from configs.settings import T

START = date(2019, 1, 5)
DATES = [START + timedelta(days=30 * i) for i in range(72)]  # ~monthly, 6 years
T_DAYS = np.array([(d - START).days for d in DATES], float)
N_PIX = 50
RNG = np.random.default_rng(0)


def seasonal(noise=0.01):
    """NDVI-like: 0.4 mean, 0.15 annual + 0.05 semi-annual amplitude, per-pixel phase jitter."""
    ph = RNG.uniform(-0.3, 0.3, N_PIX)
    w = 2 * np.pi * T_DAYS[:, None] / 365.25
    y = 0.4 + 0.15 * np.cos(w + ph) + 0.05 * np.sin(2 * w)
    return y + RNG.normal(0, noise, y.shape)


def run(y, valid=None):
    valid = np.ones_like(y, bool) if valid is None else valid
    base = harmonic.baseline_mask(T_DAYS)
    model = harmonic.fit(y, valid & base[:, None], T_DAYS)
    r = harmonic.residuals(y, valid, model, T_DAYS)
    return breaks.detect(r, valid & ~base[:, None])


def test_baseline_is_first_baseline_months():
    base = harmonic.baseline_mask(T_DAYS)
    assert base.sum() == 13  # days 0..360 of a 30-day cadence within 12 months (365.25 d)
    assert base[: base.sum()].all() and not base[base.sum() :].any()


def test_trend_only_with_long_baseline():
    assert harmonic.design(T_DAYS, trend=False).shape[1] == 5
    assert harmonic.design(T_DAYS, trend=True).shape[1] == 6
    assert not harmonic.use_trend(
        T_DAYS[harmonic.baseline_mask(T_DAYS)]
    )  # 12-month baseline < TREND_MIN_MONTHS


def test_fit_recovers_seasonal_model():
    y = seasonal(noise=0.0)
    base = harmonic.baseline_mask(T_DAYS)
    m = harmonic.fit(y, np.broadcast_to(base[:, None], y.shape), T_DAYS)
    assert np.abs(harmonic.predict(m, T_DAYS) - y).max() < 1e-6  # exact on a noise-free harmonic signal
    assert (m.sigma == T.breaks.SIGMA_MIN).all()  # MAD 0 -> floored


def test_pure_seasonality_no_break():
    b = run(seasonal())
    assert (b.idx < 0).all()


def test_step_change_breaks_at_the_right_date():
    y = seasonal()
    step_at = 40
    y[step_at:] -= 0.3  # clearance: NDVI drops and stays down
    b = run(y)
    assert (b.idx == step_at).all() and (b.sign == -1).all()
    assert (b.last_normal == step_at - 1).all()


def test_single_spike_no_break():
    y = seasonal()
    y[40] -= 0.5
    assert (run(y).idx < 0).all()


def test_two_spikes_with_a_gap_no_break():
    y = seasonal()
    y[40] -= 0.5
    y[42] -= 0.5  # obs 41 is normal: the run resets
    assert (run(y).idx < 0).all()


def test_cloudy_observations_are_ignored():
    y = seasonal()
    valid = np.ones_like(y, bool)
    y[30:33] = -0.5  # three consecutive cloud-contaminated obs...
    valid[30:33] = False  # ...masked out (q < Q_MIN): must not create a break
    assert (run(y, valid).idx < 0).all()


def test_cloud_inside_a_real_run_does_not_hide_it():
    y = seasonal()
    y[40:] -= 0.3
    valid = np.ones_like(y, bool)
    valid[41] = False  # a cloudy date inside the persistent run: skipped, not a reset
    b = run(y, valid)
    assert (b.idx == 40).all() and (b.last_normal == 39).all()


def test_step_during_masked_period_reports_first_valid_obs():
    y = seasonal()
    y[40:] -= 0.3
    valid = np.ones_like(y, bool)
    valid[38:42] = False  # change happened while cloudy: earliest *supported* obs is 42, last normal 37
    b = run(y, valid)
    assert (b.idx == 42).all() and (b.last_normal == 37).all()


def test_pixel_without_enough_baseline_is_not_modelled():
    y = seasonal()
    valid = np.ones_like(y, bool)
    valid[: T.breaks.MIN_BASELINE_OBS + 6, 0] = False  # pixel 0: baseline wiped out
    y[40:, 0] -= 0.3
    b = run(y, valid)
    assert b.idx[0] < 0 and not b.modelled[0]
    assert b.modelled[1:].all()


@pytest.mark.parametrize("persist", [T.breaks.K_PERSIST - 1])
def test_run_shorter_than_k_is_not_a_break(persist):
    y = seasonal()
    y[40 : 40 + persist] -= 0.5
    assert (run(y).idx < 0).all()


def test_cascade_block_combines_indices_and_deltas():
    from change import cascade

    n = N_PIX
    ndvi = seasonal()
    ndvi[40:, :25] -= 0.3  # half the pixels: vegetation lost at obs 40
    flat = np.full((len(T_DAYS), n), 0.1) + RNG.normal(0, 0.005, (len(T_DAYS), n))
    mndwi = np.full((len(T_DAYS), n), -0.3) + RNG.normal(0, 0.005, (len(T_DAYS), n))
    out = cascade.block({"ndvi": ndvi, "mndwi": mndwi, "ndbi": flat.copy(), "bsi": flat.copy()}, T_DAYS)
    assert (out["break_idx"][:25] == 40).all() and np.isnan(out["break_idx"][25:]).all()
    assert (out["last_normal_idx"][:25] == 39).all()
    assert np.allclose(out["d_ndvi"][:25], -0.3, atol=0.03)  # deseasonalised: the seasonal cycle is removed
    assert np.isnan(out["d_ndvi"][25:]).all()
    assert (out["water_pre"] == 0).all()
