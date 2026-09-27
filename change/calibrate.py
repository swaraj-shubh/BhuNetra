"""Isotonic calibration: raw event score -> probability the event is a real change of its type (DESIGN.md §3, M5).

Pool-adjacent-violators in numpy (no scikit-learn). Fitted only on the calibration split; applied as a logged,
versioned step (the raw score stays in the event evidence, §2.2).
"""

import json
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Calibration:
    x: tuple[float, ...]  # increasing breakpoints (raw score)
    p: tuple[float, ...]  # non-decreasing calibrated probability at each breakpoint
    version: str = "uncalibrated"

    def predict(self, raw: np.ndarray) -> np.ndarray:
        return np.asarray(np.interp(np.asarray(raw, dtype=float), self.x, self.p))

    def to_json(self) -> str:
        return json.dumps({"x": self.x, "p": self.p, "version": self.version})

    @classmethod
    def from_json(cls, s: str) -> "Calibration":
        d = json.loads(s)
        return cls(tuple(d["x"]), tuple(d["p"]), d.get("version", "uncalibrated"))


def fit_isotonic(raw: np.ndarray, label: np.ndarray, version: str = "uncalibrated") -> Calibration:
    raw, label = np.asarray(raw, dtype=float), np.asarray(label, dtype=float)
    if raw.size == 0 or raw.shape != label.shape:
        raise ValueError("need equally sized, non-empty raw scores and 0/1 labels")
    xs, inv = np.unique(raw, return_inverse=True)  # ties share one block
    sums = np.bincount(inv, weights=label)
    counts = np.bincount(inv).astype(float)
    blocks: list[list[float]] = []  # [sum, count, n_breakpoints]
    for s, c in zip(sums, counts, strict=True):
        blocks.append([s, c, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s2, c2, n2 = blocks.pop()
            blocks[-1][0] += s2
            blocks[-1][1] += c2
            blocks[-1][2] += n2
    p = np.repeat([b[0] / b[1] for b in blocks], [int(b[2]) for b in blocks])
    return Calibration(tuple(xs.tolist()), tuple(float(v) for v in p), version)
