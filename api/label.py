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


def _human() -> dict[str, dict]:
    out: dict[str, dict] = {}
    if HUMAN.exists():
        for line in HUMAN.read_text(encoding="utf-8").splitlines():
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


@router.get("/items")
def items(set: str) -> dict:
    if set not in ("calibration", "heldout"):
        raise HTTPException(422, "set must be calibration or heldout")
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
def save(event_id: str, lab: Label, user: dict = ANALYST) -> dict:
    if not EVENT_ID.match(event_id) or event_id not in {r["event_id"] for r in _csv(SAMPLE)}:
        raise HTTPException(404, "not in the M5 sample")
    if lab.true_type is not None and lab.true_type not in TYPES:
        raise HTTPException(422, f"true_type must be one of {TYPES}")
    if user["auth"]:
        lab = lab.model_copy(update={"labeller": user["name"]})
    rec = {"event_id": event_id, **lab.model_dump(), "ts": lineage.now_ts()}
    HUMAN.parent.mkdir(parents=True, exist_ok=True)
    with open(HUMAN, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec
