"""Fit the M5 event calibration on the CALIBRATION split only -> configs/calibration.json + reliability report.

    python eval/fit_calibration.py        # needs DATABASE_URL (the live events carry the evidence features)

Labels: data/labels/m5_human.jsonl (last record per event), split from eval/labels/m5_sample.csv, geometry from the
frozen snapshot eval/labels/m5_cases.geojson matched to the current events (same place, break within 45 days).
Held-out labels are never read here (DESIGN.md §2.10). Model: logistic over change/calibrate.FEATURES, then
isotonic on its score (§3). OPERATING_CONF = lowest threshold whose out-of-fold precision reaches
TARGET_PRECISION; folds are spatial blocks of 8x8 tiles so neighbouring events never straddle train/test.
"""

import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from change.calibrate import FEATURES, EventModel, event_features, fit_isotonic
from configs.settings import T, paths
from index import db

LABELS = Path(__file__).resolve().parent / "labels"
OUT = paths.root / "configs" / "calibration.json"
REPORTS = Path(__file__).resolve().parent / "reports"
GRID = np.round(np.arange(0.30, 0.96, 0.05), 2)


def dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    split = {
        r["event_id"]: r["split"] for r in csv.DictReader((LABELS / "m5_sample.csv").open(encoding="utf-8"))
    }
    lab: dict[str, dict] = {}
    for line in (paths.data / "labels" / "m5_human.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if split.get(r["event_id"]) == "calibration":
                lab[r["event_id"]] = r
    snap = json.loads((LABELS / "m5_cases.geojson").read_text(encoding="utf-8"))["features"]
    X, y, g, rows = [], [], [], []
    with db.engine().connect() as conn:
        for f in snap:
            p = f["properties"]
            lb = lab.get(p["event_id"])
            if not lb or lb["real"] not in ("yes", "no"):
                continue
            r = (
                conn.execute(
                    text(
                        "SELECT e.event_id, e.change_type::text AS change_type, e.confidence, e.evidence "
                        "FROM change_events e, (SELECT ST_SetSRID(ST_GeomFromGeoJSON(:g), 4326) AS geom) c "
                        "WHERE ST_Intersects(e.geom, c.geom) "
                        "AND abs(e.earliest_supported_date - CAST(:es AS date)) <= 45 "
                        "ORDER BY ST_Area(ST_Intersection(e.geom, c.geom)) DESC LIMIT 1"
                    ),
                    {"g": json.dumps(f["geometry"]), "es": p["earliest_supported_date"]},
                )
                .mappings()
                .first()
            )
            if r is None:
                continue
            ev = r["evidence"]
            raw = float(
                ev.get("raw_confidence", r["confidence"])
            )  # already-calibrated events keep their raw score
            X.append(event_features(raw, r["change_type"], ev))
            y.append(lb["real"] == "yes")
            _, _, row, col = p["tile_id"].split("_")
            g.append(f"{p['tile_id'].split('_')[0]}_{int(row) // 8}_{int(col) // 8}")
            rows.append({"case": p["event_id"], "event_id": r["event_id"], "type": r["change_type"]})
    return np.array(X, float), np.array(y, float), np.array(g), rows


def fit(X: np.ndarray, y: np.ndarray, version: str, operating_conf: float) -> EventModel:
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(X, y)
    iso = fit_isotonic(lr.predict_proba(X)[:, 1], y, version)
    return EventModel(
        version, tuple(float(c) for c in lr.coef_[0]), float(lr.intercept_[0]), iso, operating_conf
    )


def oof(X: np.ndarray, y: np.ndarray, g: np.ndarray) -> np.ndarray:
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups=g):
        p[te] = fit(X[tr], y[tr], "cv", 0.5).predict(X[te])
    return p


def reliability_png(p: np.ndarray, y: np.ndarray, out: Path, bins: int = 10) -> list[dict]:
    rows, S, M = [], 400, 40
    im = Image.new("RGB", (S + 2 * M, S + 2 * M), "white")
    d = ImageDraw.Draw(im)
    d.rectangle([M, M, M + S, M + S], outline="black")
    d.line([M, M + S, M + S, M], fill=(160, 160, 160))  # perfect calibration
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (p >= lo) & ((p < hi) if b < bins - 1 else (p <= hi))
        if m.sum():
            px, py = float(p[m].mean()), float(y[m].mean())
            rows.append(
                {
                    "bin": f"{lo:.1f}-{hi:.1f}",
                    "n": int(m.sum()),
                    "mean_pred": round(px, 3),
                    "frac_real": round(py, 3),
                }
            )
            cx, cy = M + px * S, M + S - py * S
            r = 3 + min(10, int(m.sum()) // 5)
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(31, 119, 180))
    d.text(
        (M, 10),
        "Reliability (out-of-fold, calibration split): x = predicted, y = fraction real",
        fill="black",
    )
    im.save(out)
    return rows


def main() -> None:
    X, y, g, _ = dataset()
    p = oof(X, y, g)
    sweep = []
    for t in GRID:
        s = p >= t
        sweep.append(
            {
                "t": float(t),
                "shown": int(s.sum()),
                "precision": float(y[s].mean()) if s.sum() else None,
                "recall": float(y[s].sum() / y.sum()),
            }
        )
    ok = [r for r in sweep if r["precision"] is not None and r["precision"] >= T.events.TARGET_PRECISION]
    op = min(r["t"] for r in ok) if ok else float(GRID[-1])
    stamp = datetime.now(UTC).date().isoformat()
    model = fit(X, y, f"cal-{stamp}", op)
    meta = {
        "n": len(y),
        "n_real": int(y.sum()),
        "groups": len(set(g)),
        "label_source": "data/labels/m5_human.jsonl, calibration split (model pre-labels, human-accepted)",
        "oof_brier": round(float(np.mean((p - y) ** 2)), 4),
        "target_precision": T.events.TARGET_PRECISION,
    }
    OUT.write_text(json.dumps(model.to_dict() | {"fit": meta}, indent=1), encoding="utf-8")
    REPORTS.mkdir(exist_ok=True)
    rel = reliability_png(p, y, REPORTS / f"{stamp}_calibration_reliability.png")
    rep = {"model": model.to_dict(), "fit": meta, "sweep": sweep, "reliability": rel}
    (REPORTS / f"{stamp}_calibration.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(
        json.dumps(
            {"operating_conf": op, **meta, "coef": dict(zip(FEATURES, np.round(model.coef, 2).tolist()))}
        )
    )


if __name__ == "__main__":
    main()
