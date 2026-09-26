"""Seasonal baseline per pixel (DESIGN.md §7). Pure numpy, vectorised over pixels (no per-pixel Python loop).

    y(t) = a0 + a1 cos(2πt/365.25) + b1 sin(2πt/365.25) + a2 cos(4πt/365.25) + b2 sin(4πt/365.25) [+ c·t]
    r(t) = (y - ŷ) / σ,   σ = max(1.4826 · MAD(baseline residuals), SIGMA_MIN)

The c·t trend is fitted only when the baseline spans at least TREND_MIN_MONTHS (M3 decision): a
trend fitted on one year and extrapolated over six creates false breaks.
Shapes: T observations x N pixels. `t` is days since the first observation.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from configs.settings import T as CFG

YEAR_D = 365.25
MONTH_D = YEAR_D / 12

F = NDArray[np.float64]
B = NDArray[np.bool_]


@dataclass(frozen=True)
class Model:
    coef: F  # (N, P); NaN rows = pixel not modelled (too few baseline observations)
    sigma: F  # (N,)
    trend: bool

    @property
    def modelled(self) -> B:
        return np.isfinite(self.sigma)


def baseline_mask(t: F) -> B:
    """True for observations within the first BASELINE_MONTHS of the series."""
    return np.asarray(t - t.min() <= CFG.breaks.BASELINE_MONTHS * MONTH_D)


def use_trend(t_baseline: F) -> bool:
    return bool(t_baseline.size and np.ptp(t_baseline) >= CFG.breaks.TREND_MIN_MONTHS * MONTH_D)


def design(t: F, trend: bool) -> F:
    w = 2 * np.pi * t / YEAR_D
    cols = [np.ones_like(t), np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)]
    if trend:
        cols.append(t / YEAR_D)  # per year: keeps the normal equations well conditioned
    return np.stack(cols, axis=1)


def fit(y: F, w: B, t: F) -> Model:
    """Weighted least squares on the observations flagged by w (baseline & valid), all pixels at once."""
    trend = use_trend(t[w.any(axis=1)])
    x = design(t, trend)  # (T, P)
    p = x.shape[1]
    wf = w.astype(np.float64)
    yz = np.where(w, y, 0.0)
    a = (wf.T @ (x[:, :, None] * x[:, None, :]).reshape(len(t), p * p)).reshape(-1, p, p)  # (N, P, P)
    b = (wf * yz).T @ x  # (N, P)
    n = w.sum(axis=0)
    ok = n >= CFG.breaks.MIN_BASELINE_OBS
    a[~ok] = np.eye(p)  # keep the batched pseudo-inverse well defined; these rows are discarded below
    coef = np.einsum("npq,nq->np", np.linalg.pinv(a), b)
    coef[~ok] = np.nan

    res = np.where(w, y - x @ coef.T, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns = unmodelled pixels
        mad = np.nanmedian(np.abs(res - np.nanmedian(res, axis=0)), axis=0)
    sigma = np.maximum(1.4826 * mad, CFG.breaks.SIGMA_MIN)
    sigma[~ok] = np.nan
    return Model(coef=coef, sigma=sigma, trend=trend)


def predict(m: Model, t: F) -> F:
    return design(t, m.trend) @ m.coef.T  # (T, N)


def residuals(y: F, valid: B, m: Model, t: F) -> F:
    """Standardised residuals; NaN where the observation is invalid or the pixel is not modelled."""
    return np.where(valid, (y - predict(m, t)) / m.sigma, np.nan)
