"""Persistent break detection (DESIGN.md §2.5, §7). Pure numpy: loops over observations, vectorised over pixels.

break = first valid monitoring obs with |r| > R_MAX followed by K_PERSIST-1 more same-sign exceedances among the
following *valid* observations (invalid/cloudy obs are skipped: they neither extend nor reset a run).
earliest_supported = first obs of that run; last_normal = last valid obs before it (baseline included).
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from configs.settings import T as CFG

I = NDArray[np.int64]


@dataclass(frozen=True)
class Breaks:
    idx: I  # (N,) observation index of earliest_supported_date, -1 = no break
    last_normal: I  # (N,) observation index of last_normal_date, -1 = none / no break
    sign: NDArray[np.int8]  # (N,) +1 / -1 direction of the residual run, 0 = no break
    modelled: NDArray[np.bool_]  # (N,) pixel had a baseline model


def detect(r: NDArray[np.float64], monitor: NDArray[np.bool_]) -> Breaks:
    """r: (T, N) standardised residuals, NaN = invalid. monitor: (T, N) or (T, 1) monitoring-period flag."""
    n_t, n = r.shape
    k, r_max = CFG.breaks.K_PERSIST, CFG.breaks.R_MAX
    monitor = np.broadcast_to(monitor, r.shape)
    idx = np.full(n, -1, np.int64)
    last_normal = np.full(n, -1, np.int64)
    sign = np.zeros(n, np.int8)
    run_sign = np.zeros(n, np.int8)
    run_len = np.zeros(n, np.int64)
    run_start = np.full(n, -1, np.int64)
    run_prev = np.full(n, -1, np.int64)
    prev_valid = np.full(n, -1, np.int64)

    for t in range(n_t):
        v = np.isfinite(r[t])
        live = v & monitor[t] & (idx < 0)
        s = np.where(np.abs(np.nan_to_num(r[t])) > r_max, np.sign(np.nan_to_num(r[t])), 0).astype(np.int8)
        cont = live & (s != 0) & (s == run_sign) & (run_len > 0)
        new = live & (s != 0) & ~cont
        reset = live & (s == 0)
        run_len[cont] += 1
        run_len[new] = 1
        run_sign[new] = s[new]
        run_start[new] = t
        run_prev[new] = prev_valid[new]
        run_len[reset] = 0
        run_sign[reset] = 0
        hit = live & (run_len >= k)
        idx[hit] = run_start[hit]
        last_normal[hit] = run_prev[hit]
        sign[hit] = run_sign[hit]
        prev_valid[v] = t

    return Breaks(
        idx=idx, last_normal=last_normal, sign=sign, modelled=np.asarray(np.isfinite(r).any(axis=0))
    )
