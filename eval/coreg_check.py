"""Measure Sentinel-2 co-registration per scene (PS 2.2.3). `python eval/coreg_check.py [aoi ...]`.

Reference = per-pixel median NIR of the clearest scenes of the AOI; every scene is phase-correlated against it
(ingest/coreg.py) in four 512 px windows with >= 90 % valid pixels, and the median window shift is the scene shift.
Writes eval/reports/<date>_coreg.json. Measurement only: q still uses shift_px = 0 until a rebuild adopts it.
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.settings import T, aois, paths
from ingest import coreg, scene

WIN, N_REF, MIN_VALID = 512, 12, 0.9


def windows(shape: tuple[int, int]) -> list[tuple[slice, slice]]:
    h, w = shape
    ys, xs = (h // 4 - WIN // 2, 3 * h // 4 - WIN // 2), (w // 4 - WIN // 2, 3 * w // 4 - WIN // 2)
    return [(slice(max(y, 0), max(y, 0) + WIN), slice(max(x, 0), max(x, 0) + WIN)) for y in ys for x in xs]


def run(aoi: str) -> dict:
    folders = sorted(p.parent for p in (paths.raw / aois[aoi].optical_source / aoi).glob("*/scene.json"))
    obs = []
    for f in folders:
        try:
            o = scene.load(f)
        except scene.SceneRejected:
            continue
        obs.append(
            (o.date, float(o.valid.mean()), np.where(o.valid, o.refl["nir"], np.nan).astype(np.float32))
        )
    clear = sorted(obs, key=lambda x: -x[1])[:N_REF]
    ref = np.nanmedian(np.stack([c[2] for c in clear]), axis=0)
    rows = []
    for date, vf, nir in obs:
        shifts = []
        for wy, wx in windows(nir.shape):
            a, b = ref[wy, wx], nir[wy, wx]
            if np.isfinite(b).mean() >= MIN_VALID and np.isfinite(a).mean() >= MIN_VALID:
                shifts.append(coreg.shift(a, b))
        if shifts:
            dy, dx = np.median(np.array(shifts), axis=0)
            rows.append(
                {
                    "date": date,
                    "valid_frac": round(vf, 3),
                    "dy": round(float(dy), 3),
                    "dx": round(float(dx), 3),
                    "shift_px": round(coreg.magnitude(dy, dx), 3),
                    "windows": len(shifts),
                }
            )
    m = np.array([r["shift_px"] for r in rows])
    return {
        "aoi": aoi,
        "scenes": len(obs),
        "measured": len(rows),
        "median_px": round(float(np.median(m)), 3),
        "p90_px": round(float(np.percentile(m, 90)), 3),
        "max_px": round(float(m.max()), 3),
        "over_accept": int((m > T.quality.SHIFT_ACCEPT_PX).sum()),
        "over_reject": int((m > T.quality.SHIFT_REJECT_PX).sum()),
        "rows": rows,
    }


if __name__ == "__main__":
    res = [run(a) for a in (sys.argv[1:] or sorted(aois))]
    out = paths.root / "eval" / "reports" / f"{datetime.now(UTC).date()}_coreg.json"
    out.write_text(
        json.dumps({"window_px": WIN, "reference_scenes": N_REF, "results": res}, indent=1), encoding="utf-8"
    )
    for r in res:
        print(json.dumps({k: v for k, v in r.items() if k != "rows"}))
    print("written", out)
