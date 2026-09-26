"""Rocchio feedback (DESIGN.md §7): q' = normalise(alpha q + beta mean(confirmed) - gamma mean(rejected))."""

import numpy as np

from configs.settings import CONFIGS, _yaml

_R = _yaml(CONFIGS / "fusion.yaml")["rocchio"]
ALPHA, BETA, GAMMA = _R["alpha"], _R["beta"], _R["gamma"]


def rocchio(q: np.ndarray, confirmed: list[np.ndarray], rejected: list[np.ndarray]) -> np.ndarray:
    out = ALPHA * np.asarray(q, dtype=np.float64)
    if confirmed:
        out = out + BETA * np.mean(confirmed, axis=0)
    if rejected:
        out = out - GAMMA * np.mean(rejected, axis=0)
    return (out / np.linalg.norm(out)).astype(np.float32)
