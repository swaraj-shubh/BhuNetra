"""M3 sanity check on OSCD (DESIGN.md §13): the change *rules* on bitemporal Sentinel-2 pairs vs urban-change labels.

OSCD gives two dates per city, not a time series, so the harmonic/persistence stages cannot run here. What this
checks is the index logic + thresholds (configs/thresholds.yaml -> classify) + the MIN_EVENT_PX sieve, pixel by pixel,
on the 10 OSCD test cities. The two dates are often in different seasons, so this is a harsh test of the rules alone.

    python -m eval.oscd_check   ->   eval/reports/<date>_oscd_check.{json,md}
"""

import io
import json
import zipfile
from datetime import UTC, datetime

import numpy as np
import rasterio
import rasterio.features

from configs.settings import T, paths
from features import indices

ROOT = paths.benchmarks / "oscd"
IMGS = ROOT / "Onera Satellite Change Detection dataset - Images.zip"
LABELS = ROOT / "Onera Satellite Change Detection dataset - Test Labels.zip"
BANDS = {"blue": "B02", "green": "B03", "red": "B04", "nir": "B08", "swir1": "B11"}
REPORTS = paths.root / "eval" / "reports"


def _read(z: zipfile.ZipFile, name: str) -> np.ndarray:
    with rasterio.MemoryFile(io.BytesIO(z.read(name))) as mf, mf.open() as d:
        return d.read(1).astype(np.float32)


def _ix(z: zipfile.ZipFile, city: str, k: int) -> dict[str, np.ndarray]:
    base = f"Onera Satellite Change Detection dataset - Images/{city}/imgs_{k}_rect/"
    refl = {
        c: _read(z, base + f"{b}.tif") for c, b in BANDS.items()
    }  # scale cancels in normalised differences
    return indices.compute(refl)


def predict(i1: dict[str, np.ndarray], i2: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    c = T.classify
    d = {n: i2[n] - i1[n] for n in indices.NAMES}
    construction = (d["ndvi"] <= c.CONS_DNDVI) & (d["ndbi"] >= c.CONS_DNDBI)
    clearance = (d["ndvi"] <= c.CLR_DNDVI) & (d["bsi"] >= c.CLR_DBSI)
    out = {"construction_rule": construction, "construction_or_clearance": construction | clearance}
    return {
        k: rasterio.features.sieve(v.astype(np.uint8), size=T.events.MIN_EVENT_PX, connectivity=8).astype(
            bool
        )
        for k, v in out.items()
    }


def prf(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0.0,
    }


def main() -> None:
    zi, zl = zipfile.ZipFile(IMGS), zipfile.ZipFile(LABELS)
    cities = sorted({n.split("/")[1] for n in zl.namelist() if n.endswith("-cm.tif")})
    totals: dict[str, dict[str, int]] = {}
    per_city = {}
    prevalence = []
    for city in cities:
        cm = _read(zl, f"Onera Satellite Change Detection dataset - Test Labels/{city}/cm/{city}-cm.tif")
        # README says 0/1, but the shipped test tifs are 1 = no change, 2 = change: take the higher value as change
        assert len(np.unique(cm)) <= 2, (city, np.unique(cm))
        truth = cm == cm.max()
        i1, i2 = _ix(zi, city, 1), _ix(zi, city, 2)
        h, w = truth.shape
        i1 = {k: v[:h, :w] for k, v in i1.items()}
        i2 = {k: v[:h, :w] for k, v in i2.items()}
        prevalence.append(float(truth.mean()))
        per_city[city] = {}
        for variant, pred in predict(i1, i2).items():
            m = prf(pred, truth)
            per_city[city][variant] = m
            t = totals.setdefault(variant, {"tp": 0, "fp": 0, "fn": 0})
            for k in t:
                t[k] += m[k]
    overall = {}
    for variant, t in totals.items():
        p = t["tp"] / max(t["tp"] + t["fp"], 1)
        r = t["tp"] / max(t["tp"] + t["fn"], 1)
        overall[variant] = {**t, "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}

    REPORTS.mkdir(parents=True, exist_ok=True)
    stem = REPORTS / f"{datetime.now(UTC).date()}_oscd_check"
    meta = {
        "cities": cities,
        "thresholds": T.classify.model_dump(),
        "min_event_px": T.events.MIN_EVENT_PX,
        "mean_change_prevalence": float(np.mean(prevalence)),
    }
    stem.with_suffix(".json").write_text(
        json.dumps({"meta": meta, "overall": overall, "per_city": per_city}, indent=1), encoding="utf-8"
    )
    lines = ["| variant | precision | recall | F1 |", "|---|---|---|---|"]
    lines += [
        f"| {v} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |" for v, m in overall.items()
    ]
    lines.append(f"| (all-change baseline) | {np.mean(prevalence):.3f} | 1.000 | - |")
    stem.with_suffix(".md").write_text(
        "# M3 sanity check: change rules on OSCD test pairs (pixel level)\n\n"
        "Bitemporal pairs, so no harmonic/persistence stage: this checks the index rules, thresholds and "
        "MIN_EVENT_PX sieve only.\n\n" + "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
