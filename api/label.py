"""M5 labelling tool (evaluation only, not part of the DESIGN.md §12 contract). Open http://localhost:8000/label

Calibration items show the model pre-label for correction; held-out items are labelled blind (no pre-label, no system
type/confidence in the view). Labels append to data/labels/m5_human.jsonl (latest entry per event wins).
"""

import csv
import json
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from api.auth import ANALYST
from configs.settings import paths
from governance import lineage

router = APIRouter(prefix="/label", tags=["labelling (eval only)"])
SAMPLE = paths.root / "eval" / "labels" / "m5_sample.csv"
PRELABELS = paths.root / "eval" / "labels" / "m5_prelabels.csv"
HUMAN = paths.data / "labels" / "m5_human.jsonl"
BLIND = paths.data / "labels" / "m5_human_blind.jsonl"  # calibration recheck: labelled without the pre-label
RECHECK_N, RECHECK_SEED = 100, 20260928  # fixed sample, chosen before any recheck label exists
VIEWS = paths.data / "labels" / "m5_views"
TYPES = ("construction", "clearance", "water", "road", "other")
EVENT_ID = re.compile(r"^CHG-\d{6}$")


class Label(BaseModel):
    real: str = Field(pattern="^(yes|no|unsure)$")
    true_type: str | None = None  # what the change really is (None when real != yes)
    note: str = Field(default="", max_length=500)
    labeller: str = Field(min_length=1, max_length=60)


def _csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _human(path: Path | None = None) -> dict[str, dict]:
    path = path or HUMAN  # resolved per call (tests and tools repoint HUMAN)
    out: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["event_id"]] = r
    return out


@router.get("")
def page() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "label.html")


@router.get("/guidelines")
def guidelines() -> FileResponse:
    return FileResponse(
        paths.root / "eval" / "labels" / "M5_GUIDELINES.md", media_type="text/plain; charset=utf-8"
    )


def recheck_ids() -> list[str]:
    """The blind recheck sample: RECHECK_N calibration events, seeded (independent of any label)."""
    import random

    ids = sorted(r["event_id"] for r in _csv(SAMPLE) if r["split"] == "calibration")
    return sorted(random.Random(RECHECK_SEED).sample(ids, min(RECHECK_N, len(ids))))


@router.get("/items")
def items(set: str) -> dict:
    if set == "recheck":  # blind like held-out: no pre-label, no system type, no earlier label
        blind = _human(BLIND)
        out = [{"event_id": e, "human": blind.get(e)} for e in recheck_ids()]
        return {"set": set, "items": out, "done": sum(i["human"] is not None for i in out)}
    if set not in ("calibration", "heldout"):
        raise HTTPException(422, "set must be calibration, heldout or recheck")
    pre = {r["event_id"]: r for r in _csv(PRELABELS)} if set == "calibration" else {}
    human = _human()
    out = []
    for r in _csv(SAMPLE):
        if r["split"] != set:
            continue
        item = {"event_id": r["event_id"], "human": human.get(r["event_id"])}
        if set == "calibration":  # held-out stays blind: no system type, no pre-label
            item["system_type"] = r["change_type"]
            p = pre.get(r["event_id"])
            item["prelabel"] = (
                {"real": p["real"], "true_type": p["true_type"] or None, "note": p["note"]} if p else None
            )
        out.append(item)
    return {"set": set, "items": out, "done": sum(i["human"] is not None for i in out)}


@router.get("/view/{event_id}.png")
def view(event_id: str) -> FileResponse:
    f = VIEWS / f"{event_id}.png"
    if not EVENT_ID.match(event_id) or not f.exists():
        raise HTTPException(404)
    return FileResponse(f)


@router.post("/{event_id}")
def save(event_id: str, lab: Label, blind: bool = False, user: dict = ANALYST) -> dict:
    if not EVENT_ID.match(event_id) or event_id not in {r["event_id"] for r in _csv(SAMPLE)}:
        raise HTTPException(404, "not in the M5 sample")
    if lab.true_type is not None and lab.true_type not in TYPES:
        raise HTTPException(422, f"true_type must be one of {TYPES}")
    if user["auth"]:
        lab = lab.model_copy(update={"labeller": user["name"]})
    if blind and event_id not in recheck_ids():
        raise HTTPException(404, "not in the blind recheck sample")
    rec = {"event_id": event_id, **lab.model_dump(), "ts": lineage.now_ts()} | (
        {"blind": True} if blind else {}
    )
    target = BLIND if blind else HUMAN
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


# Retrieval judgements (eval/retrieval_pool.py): each pooled site is judged blind against its query text.
POOL = paths.data / "labels" / "retrieval_pool.json"
JUDGED = paths.data / "labels" / "retrieval_judgements.jsonl"


class Judgement(BaseModel):
    qid: str = Field(pattern=r"^q\d{2,3}$")
    tile_id: str = Field(min_length=3, max_length=80)
    relevant: str = Field(pattern="^(yes|no|unsure)$")
    labeller: str = Field(min_length=1, max_length=60)


@router.get("/retrieval")
def retrieval_page() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "label_retrieval.html")


@router.get("/retrieval/items")
def retrieval_items() -> dict:
    """Pooled sites per query in the fixed shuffled order: no rank, score or system reasoning (blind)."""
    if not POOL.exists():
        raise HTTPException(404, "no pool yet: python eval/retrieval_pool.py build")
    done: dict[tuple[str, str], str] = {}
    if JUDGED.exists():
        for line in JUDGED.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["qid"], r["tile_id"])] = r["relevant"]
    pool = json.loads(POOL.read_text(encoding="utf-8"))["pool"]
    out = []
    for q in pool:
        sites = [q["sites"][i] for i in q["shown_order"]]
        out.append(
            {
                "qid": q["id"],
                "query": q["query"],
                "aoi": q["aoi"],
                "sites": [
                    {
                        "tile_id": s["tile_id"],
                        "aoi": s["aoi"],
                        "chip_url": s["chip_url"],
                        "event_ids": s["event_ids"],
                        "judged": done.get((q["id"], s["tile_id"])),
                    }
                    for s in sites
                ],
            }
        )
    n = sum(len(q["sites"]) for q in out)
    return {
        "queries": out,
        "sites": n,
        "judged": sum(s["judged"] is not None for q in out for s in q["sites"]),
    }


@router.post("/retrieval/judge")
def retrieval_judge(j: Judgement, user: dict = ANALYST) -> dict:
    pool = json.loads(POOL.read_text(encoding="utf-8"))["pool"] if POOL.exists() else []
    if not any(q["id"] == j.qid and any(s["tile_id"] == j.tile_id for s in q["sites"]) for q in pool):
        raise HTTPException(404, "not in the retrieval pool")
    if user["auth"]:
        j = j.model_copy(update={"labeller": user["name"]})
    rec = {**j.model_dump(), "ts": lineage.now_ts()}
    JUDGED.parent.mkdir(parents=True, exist_ok=True)
    with open(JUDGED, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec
