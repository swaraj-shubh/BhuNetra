"""Blind human retrieval judgements (PS 2.3: retrieval is scored on held-out queries with relevance judgements).

    python eval/retrieval_pool.py build [--api URL]   # run eval/labels/retrieval_queries.json once -> data/labels/retrieval_pool.json
    python eval/retrieval_pool.py score               # judgements -> eval/reports/<date>_retrieval_human.json (P@10)

The pool freezes what the system returned (top 10 per query, through the public API only). Judging happens in the
browser at /label/retrieval: shuffled, no rank or score shown. Judgements are keyed by (query id, tile_id), so a
rebuilt pool only needs the new sites judged. Queries were fixed before any result was viewed.
"""

import argparse
import json
import os
import random
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUERIES = ROOT / "eval" / "labels" / "retrieval_queries.json"
DATA = Path(os.environ.get("SIH_ROOT", ROOT)) / "data" / "labels"
POOL = DATA / "retrieval_pool.json"
JUDGED = DATA / "retrieval_judgements.jsonl"
REPORTS = ROOT / "eval" / "reports"
K = 10


def _post(api: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(
        api + path, json.dumps(body).encode(), {"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def build(api: str) -> dict:
    qs = json.loads(QUERIES.read_text(encoding="utf-8"))["queries"]
    pool = []
    for q in qs:
        out = _post(api, "/search/text", {"query": q["query"], "aoi": q["aoi"], "k": K})
        sites = [
            {
                "tile_id": s["tile_id"],
                "aoi": s["aoi"],
                "rank": s["rank"],
                "chip_url": s["chip_url"],
                "event_ids": sorted({e["event_id"] for e in s["events"]}),
                "geometry": s["geometry"],
            }
            for s in out["sites"]
        ]
        order = list(range(len(sites)))
        random.Random(q["id"]).shuffle(order)  # judges see a fixed shuffle, never the rank
        pool.append({**q, "parsed": out["parsed"], "sites": sites, "shown_order": order})
    doc = {"built_ts": datetime.now(UTC).isoformat(timespec="seconds"), "api": api, "k": K, "pool": pool}
    DATA.mkdir(parents=True, exist_ok=True)
    POOL.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return doc


def judgements() -> dict[tuple[str, str], dict]:
    """Latest judgement per (query id, tile_id)."""
    out: dict[tuple[str, str], dict] = {}
    if JUDGED.exists():
        for line in JUDGED.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[(r["qid"], r["tile_id"])] = r
    return out


def precision_at_k(sites: list[dict], judged: dict[tuple[str, str], dict], qid: str) -> dict:
    """P@k over judged yes/no among the top k (unsure and unjudged are left out and counted)."""
    marks = [judged.get((qid, s["tile_id"]), {}).get("relevant") for s in sites[:K]]
    yes, no = marks.count("yes"), marks.count("no")
    return {
        "p_at_k": round(yes / (yes + no), 3) if yes + no else None,
        "relevant": yes,
        "not_relevant": no,
        "unsure": marks.count("unsure"),
        "unjudged": marks.count(None),
    }


def score() -> dict:
    doc = json.loads(POOL.read_text(encoding="utf-8"))
    judged = judgements()
    per = [
        {"id": q["id"], "query": q["query"], "aoi": q["aoi"], **precision_at_k(q["sites"], judged, q["id"])}
        for q in doc["pool"]
    ]
    done = [p["p_at_k"] for p in per if p["p_at_k"] is not None]
    res = {
        "pool_built": doc["built_ts"],
        "k": K,
        "queries": len(per),
        "queries_judged": len(done),
        "mean_p_at_k": round(sum(done) / len(done), 3) if done else None,
        "per_aoi": {
            a: round(sum(v) / len(v), 3)
            for a in sorted({p["aoi"] or "all" for p in per})
            if (v := [p["p_at_k"] for p in per if (p["aoi"] or "all") == a and p["p_at_k"] is not None])
        },
        "per_query": per,
    }
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / f"{datetime.now(UTC).date()}_retrieval_human.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res | {"written": str(out)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "score"])
    ap.add_argument("--api", default=os.environ.get("SIH_API", "http://127.0.0.1:8000"))
    a = ap.parse_args()
    r = build(a.api) if a.cmd == "build" else score()
    print(json.dumps({k: v for k, v in r.items() if k not in ("pool", "per_query")}))
