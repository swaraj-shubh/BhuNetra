"""M1 CLIP go/no-go, part 2: text->image P@10 on 200 of our own 640 m composite tiles (DESIGN.md §13).

    python -m eval.own_tiles sample   # seeded, stratified by AOI, non-monsoon, non-overlapping -> eval/labels/
    python -m eval.own_tiles sheets   # numbered contact sheets (label these BEFORE running bench)
    python -m eval.own_tiles bench    # every text_vec candidate -> eval/reports/<date>_bench_own_tiles.{json,md}

Chips are made exactly as the ingest worker makes them (composite window -> rgb_quicklook -> model preprocess).
Labels: eval/labels/m1_own_tiles.csv, multi-label from CLASSES (';'-separated), with the labeller recorded.
"""

import argparse
import csv
import json
import random
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import rasterio
import torch
from PIL import Image, ImageDraw

from configs.settings import T, models, paths
from eval.bench_clip import load_candidate
from features.embed_clip import DEVICE
from ingest import tiling
from ingest.standardise import rgb_quicklook

LABELS = paths.root / "eval" / "labels" / "m1_own_tiles.csv"
REPORTS = paths.root / "eval" / "reports"
SEED, N, PER_SHEET = 26227, 200, 25
SKIP_SEASONS = ("JJAS",)  # monsoon composites are patchy
# class -> query text (fed through the deployed templates in configs/models.yaml)
CLASSES = {
    "river": "a river",
    "water_body": "a lake or reservoir",
    "sea": "the sea or a coastline",
    "wetland": "wetland or mudflats",
    "salt_pan": "salt pans",
    "farmland": "farmland",
    "forest": "forest",
    "barren": "barren land",
    "built_up": "a town or residential area",
    "industrial": "an industrial area",
    "road": "a highway or road",
    "mining": "an open-pit mine",
}
MIN_POS = 3  # classes with fewer positives are reported but left out of the mean


def _tile(tile_id: str) -> tiling.Tile:
    from configs.settings import aois

    aoi = tile_id.split("_")[0]
    epsg = tiling.aoi_epsg(aois[aoi].bbox)
    return next(
        t for t in tiling.tiles(aoi, tiling.aoi_bounds_m(aois[aoi].bbox, epsg), epsg) if t.tile_id == tile_id
    )


def chip(tile_id: str, season: str) -> np.ndarray:
    aoi = tile_id.split("_")[0]
    with rasterio.open(paths.composites / aoi / f"{season}.tif") as d:
        w = tiling.window(_tile(tile_id), d.transform)
        r, g, b = (d.read(d.descriptions.index(n) + 1, window=w) for n in ("red", "green", "blue"))
    return rgb_quicklook(r, g, b)


def sample() -> None:
    if LABELS.exists():
        raise SystemExit(f"{LABELS} exists: the sample is frozen once drawn")
    rng = random.Random(SEED)
    by_aoi: dict[str, list[tuple[str, str]]] = {}
    for side_f in sorted(paths.composites.glob("*/*.json")):
        side = json.loads(side_f.read_text(encoding="utf-8"))
        if side["season"].endswith(SKIP_SEASONS):
            continue
        st = side["tiles"]
        for tid, vf, n in zip(st["tile_id"], st["valid_frac"], st["n_obs"], strict=True):
            if vf >= T.quality.VALID_FRAC_EMB and n > 0:
                by_aoi.setdefault(side["aoi"], []).append((tid, side["season"]))
    quota = {a: N // len(by_aoi) + (i < N % len(by_aoi)) for i, a in enumerate(sorted(by_aoi))}
    rows, taken = [], []
    for aoi in sorted(by_aoi):
        pool = by_aoi[aoi]
        rng.shuffle(pool)
        got = 0
        for tid, season in pool:
            r, c = map(int, tid.split("_")[-2:])
            if any(a == aoi and abs(r - rr) < 2 and abs(c - cc) < 2 for a, rr, cc in taken):
                continue  # tiles overlap by half a tile: keep the sample free of near-duplicates
            taken.append((aoi, r, c))
            rows.append({"id": len(rows), "tile_id": tid, "season": season, "labels": "", "labeller": ""})
            got += 1
            if got == quota[aoi]:
                break
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    with open(LABELS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(
        json.dumps(
            {
                "sampled": len(rows),
                "per_aoi": {a: sum(r["tile_id"].startswith(a) for r in rows) for a in by_aoi},
            }
        )
    )


def read_labels() -> list[dict]:
    with open(LABELS, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sheets(out_dir: Path, scale: int = 3) -> None:
    rows = read_labels()
    out_dir.mkdir(parents=True, exist_ok=True)
    px = 64 * scale
    for s in range(0, len(rows), PER_SHEET):
        part = rows[s : s + PER_SHEET]
        sheet = Image.new("RGB", (5 * (px + 4), 5 * (px + 18)), "white")
        d = ImageDraw.Draw(sheet)
        for i, r in enumerate(part):
            x, y = (i % 5) * (px + 4), (i // 5) * (px + 18)
            im = Image.fromarray(chip(r["tile_id"], r["season"])).resize((px, px), Image.Resampling.NEAREST)
            sheet.paste(im, (x, y + 16))
            d.text((x + 2, y + 2), f"#{r['id']}", fill="black")
        sheet.save(out_dir / f"sheet_{s // PER_SHEET:02d}.png")
    print(json.dumps({"sheets": (len(rows) + PER_SHEET - 1) // PER_SHEET, "dir": str(out_dir)}))


@torch.no_grad()
def bench() -> None:
    rows = read_labels()
    if any(not r["labeller"] for r in rows):
        raise SystemExit("label every tile first")
    names = list(CLASSES)
    y = np.array([[c in r["labels"].split(";") for c in names] for r in rows])
    chips = [Image.fromarray(chip(r["tile_id"], r["season"])) for r in rows]
    results = []
    for name, spec in models["text_vec_candidates"].items():
        model, preprocess, tokenizer = load_candidate(name, spec)
        img = (
            torch.nn.functional.normalize(
                model.encode_image(torch.stack([preprocess(c) for c in chips]).to(DEVICE)), dim=-1
            )
            .cpu()
            .numpy()
        )
        per_class = {}
        for j, c in enumerate(names):
            e = torch.nn.functional.normalize(
                model.encode_text(
                    tokenizer([t.format(CLASSES[c]) for t in models["text_vec"]["templates"]]).to(DEVICE)
                ),
                dim=-1,
            )
            order = np.argsort(-(img @ torch.nn.functional.normalize(e.mean(0), dim=-1).cpu().numpy()))
            rel = y[order, j]
            ap = float((np.cumsum(rel)[rel] / (np.flatnonzero(rel) + 1)).mean()) if rel.any() else 0.0
            per_class[c] = {"n_pos": int(y[:, j].sum()), "P@10": float(rel[:10].mean()), "AP": ap}
        scored = [v for v in per_class.values() if v["n_pos"] >= MIN_POS]
        results.append(
            {
                "model": name,
                "P@10": float(np.mean([v["P@10"] for v in scored])),
                "mAP": float(np.mean([v["AP"] for v in scored])),
                "classes_scored": len(scored),
                "per_class": per_class,
            }
        )
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    REPORTS.mkdir(parents=True, exist_ok=True)
    stem = REPORTS / f"{datetime.now(UTC).date()}_bench_own_tiles"
    meta = {
        "n_tiles": len(rows),
        "labellers": sorted({r["labeller"] for r in rows}),
        "min_pos": MIN_POS,
        "classes": CLASSES,
        "templates": models["text_vec"]["templates"],
        "seed": SEED,
    }
    stem.with_suffix(".json").write_text(
        json.dumps({"meta": meta, "results": results}, indent=1), encoding="utf-8"
    )
    scored_classes = [c for c in names if results[0]["per_class"][c]["n_pos"] >= MIN_POS]
    head = (
        "| model | P@10 | mAP | "
        + " | ".join(f"{c} P@10 (n={results[0]['per_class'][c]['n_pos']})" for c in scored_classes)
        + " |"
    )
    lines = [head, "|---" * (3 + len(scored_classes)) + "|"]
    for r in results:
        cells = [f"{r['per_class'][c]['P@10']:.1f}" for c in scored_classes]
        lines.append(f"| {r['model']} | {r['P@10']:.3f} | {r['mAP']:.3f} | " + " | ".join(cells) + " |")
    stem.with_suffix(".md").write_text(
        f"# M1 part 2: text->image on {len(rows)} own 640 m tiles\n\nLabellers: {', '.join(meta['labellers'])}. "
        f"Classes with < {MIN_POS} positives are excluded from the means.\n\n" + "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "sheets", "bench"])
    ap.add_argument("--out", type=Path, default=paths.data / "cache" / "m1_sheets")
    a = ap.parse_args()
    {"sample": sample, "sheets": lambda: sheets(a.out), "bench": bench}[a.cmd]()
