"""M10 evaluation harness: score the running API on retrieval queries and change/no-change cases.

`python eval/run.py --queries Q.json --cases C.geojson [--api http://127.0.0.1:8000] [--k 10]` -> eval/reports/.
Uses only the public contract (DESIGN.md §12: /search/text, /change), so the organisers' hidden sets run as-is.
Matching is by geometry (EPSG:4326), never by our tile ids, because hidden judgements will not know them.

queries: {"queries": [{"id", "query", "aoi"?, "date_range"?: [t1, t2], "relevant": [GeoJSON geometry, ...]}]}
cases:   GeoJSON FeatureCollection; properties {"case_id", "label": "change"|"no_change", "t1", "t2",
         "type"?: change type, "earliest_date"?: ISO date}
A case is predicted "change" when a stored event at/above the operating confidence intersects it with
earliest_supported_date in [t1, t2]. Our own frozen held-out split is the only place our numbers come from
(§2.10); never point this at hidden data during development.
"""

import argparse
import json
import math
import os
import statistics
import time
import urllib.parse
import urllib.request
from datetime import UTC, date, datetime
from pathlib import Path

from shapely.geometry import shape

REPORTS = Path(__file__).resolve().parent / "reports"


# ---------- pure metrics ----------
def ranked_hits(site_geoms: list, relevant: list) -> list[int | None]:
    """Per ranked site: index of the first not-yet-credited relevant geometry it hits, else None."""
    used: set[int] = set()
    out: list[int | None] = []
    for g in site_geoms:
        hit = next((i for i, r in enumerate(relevant) if i not in used and g.intersects(r)), None)
        if hit is not None:
            used.add(hit)
        out.append(hit)
    return out


def retrieval_metrics(hits: list[int | None], n_relevant: int, k: int) -> dict:
    top = [h is not None for h in hits[:k]]
    ap = (
        sum(sum(top[: i + 1]) / (i + 1) for i, t in enumerate(top) if t) / min(n_relevant, k)
        if n_relevant
        else 0.0
    )
    dcg = sum(1 / math.log2(i + 2) for i, t in enumerate(top) if t)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(n_relevant, k)))
    return {
        f"p@{k}": sum(top) / k,
        f"recall@{k}": sum(top) / n_relevant if n_relevant else 0.0,
        f"ap@{k}": ap,
        f"ndcg@{k}": dcg / idcg if idcg else 0.0,
    }


def predict_case(case_geom, t1: str, t2: str, events: list[dict]) -> dict | None:
    """Highest-confidence event intersecting the case inside [t1, t2] (events already confidence-filtered)."""
    m = [
        e
        for e in events
        if t1 <= e["earliest_supported_date"] <= t2 and case_geom.intersects(shape(e["geometry"]))
    ]
    return max(m, key=lambda e: e["confidence"], default=None)


def change_metrics(rows: list[dict]) -> dict:
    """rows: {"label", "pred": event|None, "type"?, "earliest_date"?}."""
    tp = sum(r["label"] == "change" and r["pred"] is not None for r in rows)
    fp = sum(r["label"] == "no_change" and r["pred"] is not None for r in rows)
    fn = sum(r["label"] == "change" and r["pred"] is None for r in rows)
    tn = sum(r["label"] == "no_change" and r["pred"] is None for r in rows)
    p = tp / (tp + fp) if tp + fp else None
    rc = tp / (tp + fn) if tp + fn else None
    typed = [r for r in rows if r["label"] == "change" and r["pred"] and r.get("type")]
    dated = [r for r in rows if r["label"] == "change" and r["pred"] and r.get("earliest_date")]
    errs = [
        abs(
            (
                date.fromisoformat(r["pred"]["earliest_supported_date"])
                - date.fromisoformat(r["earliest_date"])
            ).days
        )
        for r in dated
    ]
    return {
        "n": len(rows),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": p,
        "recall": rc,
        "f1": 2 * p * rc / (p + rc) if p and rc else None,
        "false_alarm_rate": fp / (fp + tn) if fp + tn else None,
        "type_accuracy": sum(r["pred"]["change_type"] == r["type"] for r in typed) / len(typed)
        if typed
        else None,
        "earliest_date_abs_err_days": {"median": statistics.median(errs), "max": max(errs), "n": len(errs)}
        if errs
        else None,
    }


# ---------- API I/O ----------
def _call(api: str, path: str, body: dict | None = None) -> tuple[dict, float]:
    req = urllib.request.Request(
        api + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={
            "Content-Type": "application/json",
            **({"X-API-Key": k} if (k := os.environ.get("SIH_API_KEY")) else {}),
        },
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.loads(r.read())
    return out, (time.perf_counter() - t0) * 1000


def run_queries(api: str, spec: dict, k: int) -> dict:
    per, lat = [], []
    for q in spec["queries"]:
        body = {"query": q["query"], "k": k} | {f: q[f] for f in ("aoi", "date_range") if q.get(f)}
        out, ms = _call(api, "/search/text", body)
        lat.append(ms)
        rel = [shape(g) for g in q["relevant"]]
        hits = ranked_hits([shape(s["geometry"]) for s in out["sites"]], rel)
        per.append(
            {
                "id": q["id"],
                "query": q["query"],
                "latency_ms": round(ms, 1),
                **retrieval_metrics(hits, len(rel), k),
            }
        )
    keys = [c for c in per[0] if c not in ("id", "query", "latency_ms")] if per else []
    return {
        "k": k,
        "n_queries": len(per),
        "mean": {c: statistics.fmean(p[c] for p in per) for c in keys},
        "latency_ms": _lat(lat),
        "per_query": per,
    }


def run_cases(api: str, fc: dict) -> dict:
    cache: dict[tuple[str, str], list[dict]] = {}
    rows, lat = [], []
    for f in fc["features"]:
        pr = f["properties"]
        key = (pr["t1"], pr["t2"])
        if key not in cache:
            out, ms = _call(api, "/change?" + urllib.parse.urlencode({"t1": pr["t1"], "t2": pr["t2"]}))
            cache[key] = out["events"]
            lat.append(ms)
        pred = predict_case(shape(f["geometry"]), pr["t1"], pr["t2"], cache[key])
        rows.append({**pr, "pred": pred})
    by_type: dict[str, list[dict]] = {}
    for r in rows:  # per-class (DESIGN.md §16): no-change cases count against every class's precision
        if r["label"] == "change" and r.get("type"):
            by_type.setdefault(r["type"], []).append(r)
    neg = [r for r in rows if r["label"] == "no_change"]
    return {
        "overall": change_metrics(rows),
        "per_type": {
            # only negatives predicted as t are included, so a per-type false-alarm rate would always be 1
            t: change_metrics(v + [n for n in neg if n["pred"] and n["pred"]["change_type"] == t])
            | {"false_alarm_rate": None}
            for t, v in by_type.items()
        },
        "latency_ms": _lat(lat),
        "cases": [
            {
                "case_id": r.get("case_id"),
                "label": r["label"],
                "pred_event": r["pred"]["event_id"] if r["pred"] else None,
            }
            for r in rows
        ],
    }


def _lat(ms: list[float]) -> dict | None:
    if not ms:
        return None
    s = sorted(ms)
    return {
        "p50": round(statistics.median(s), 1),
        "p95": round(s[min(len(s) - 1, int(0.95 * len(s)))], 1),
        "n": len(s),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=os.environ.get("SIH_API", "http://127.0.0.1:8000"))
    ap.add_argument("--queries", type=Path)
    ap.add_argument("--cases", type=Path)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--name", default="eval")
    a = ap.parse_args()
    res: dict = {"api": a.api, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if a.queries:
        res["retrieval"] = run_queries(a.api, json.loads(a.queries.read_text(encoding="utf-8")), a.k)
    if a.cases:
        res["change"] = run_cases(a.api, json.loads(a.cases.read_text(encoding="utf-8")))
    out = REPORTS / f"{datetime.now(UTC).date()}_{a.name}.json"
    out.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(
        json.dumps(
            {k: v for k, v in res.items() if k in ("retrieval", "change")} | {"written": str(out)},
            default=str,
        )[:3000]
    )
