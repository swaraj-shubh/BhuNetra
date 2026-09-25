"""M1 CLIP go/no-go, part 1: zero-shot EuroSAT (RGB test split) for every text_vec candidate (DESIGN.md §13).

EuroSAT chips are 64x64 px at 10 m = 640 m, i.e. exactly one tile (TILE_M), so this also exercises the deployed
64 -> 224 px resampling (each model's own preprocess). Part 2 (P@10 on 200 own composite tiles) needs M2 composites.

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python eval/bench_clip.py [--only NAME] [--limit N]
Writes eval/reports/<date>_bench_clip.{json,md}.
"""

import argparse
import gc
import io
import json
import os
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import open_clip
import psutil
import torch
from PIL import Image

from configs.settings import models, paths
from features.embed_clip import DEVICE

EUROSAT = paths.benchmarks / "eurosat"
# Class names as used by clip_benchmark / the CLIP paper for EuroSAT.
CLASSNAMES = {
    "AnnualCrop": "annual crop land",
    "Forest": "forest",
    "HerbaceousVegetation": "brushland or shrubland",
    "Highway": "highway or road",
    "Industrial": "industrial buildings or commercial buildings",
    "Pasture": "pasture land",
    "PermanentCrop": "permanent crop land",
    "Residential": "residential buildings or homes or apartments",
    "River": "river",
    "SeaLake": "lake or sea",
}
CLIP_EUROSAT_TEMPLATES = [
    "a centered satellite photo of {}.",
    "a centered satellite photo of a {}.",
    "a centered satellite photo of the {}.",
]
REPORTS = paths.root / "eval" / "reports"


def load_candidate(name: str, spec: dict):
    """Build (model, preprocess, tokenizer) from local files only (DESIGN.md §2.8)."""
    d = paths.models / name
    if "arch" in spec:
        model, _, preprocess = open_clip.create_model_and_transforms(
            spec["arch"], pretrained=str(d / spec["file"])
        )
        return model.to(DEVICE).eval(), preprocess, open_clip.get_tokenizer(spec["arch"])
    # open_clip JSON config whose HF text tower must point at a local dir instead of the hub
    cfg = json.loads((d / spec["open_clip_config"]).read_text(encoding="utf-8"))
    text_dir = str(paths.models / spec["hf_text_dir"])
    cfg["model_cfg"]["text_cfg"]["hf_model_name"] = text_dir
    cfg["model_cfg"]["text_cfg"]["hf_tokenizer_name"] = text_dir
    local_name = f"{name}-local"
    cfg_path = paths.data / "cache" / f"{local_name}.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg["model_cfg"]), encoding="utf-8")
    open_clip.add_model_config(cfg_path)
    model, _, preprocess = open_clip.create_model_and_transforms(
        local_name, pretrained=str(d / spec["file"]), pretrained_hf=False
    )
    return model.to(DEVICE).eval(), preprocess, open_clip.get_tokenizer(local_name)


def test_split(limit: int | None) -> list[tuple[str, int]]:
    classes = list(CLASSNAMES)
    files = (EUROSAT / "eurosat-test.txt").read_text(encoding="utf-8").split()
    items = [(f, classes.index(f.rsplit("_", 1)[0])) for f in files]
    if limit:  # deterministic class-balanced subset for quick runs
        per = max(1, limit // len(classes))
        items = [it for c in range(len(classes)) for it in [x for x in items if x[1] == c][:per]]
    return items


@torch.no_grad()
def encode_images(
    model, preprocess, items, batch: int = 256 if DEVICE == "cuda" else 64
) -> tuple[np.ndarray, float]:
    zf = zipfile.ZipFile(EUROSAT / "EuroSAT.zip")
    by_name = {Path(n).name: n for n in zf.namelist() if n.endswith(".jpg")}
    out, t0 = [], time.perf_counter()
    for i in range(0, len(items), batch):
        ims = [Image.open(io.BytesIO(zf.read(by_name[f]))).convert("RGB") for f, _ in items[i : i + batch]]
        out.append(
            torch.nn.functional.normalize(
                model.encode_image(torch.stack([preprocess(im) for im in ims]).to(DEVICE)), dim=-1
            ).cpu()
        )
    return torch.cat(out).numpy(), time.perf_counter() - t0


@torch.no_grad()
def encode_classes(model, tokenizer, templates: list[str]) -> np.ndarray:
    embs = []
    for cname in CLASSNAMES.values():
        e = torch.nn.functional.normalize(
            model.encode_text(tokenizer([t.format(cname) for t in templates]).to(DEVICE)), dim=-1
        )
        embs.append(torch.nn.functional.normalize(e.mean(0), dim=-1).cpu())
    return torch.stack(embs).numpy()


def retrieval(img: np.ndarray, txt: np.ndarray, labels: np.ndarray, k: int = 10) -> tuple[float, float]:
    """Text->image: per class, rank all images by similarity. Mean P@k and mean AP over classes."""
    p_at_k, aps = [], []
    for c in range(txt.shape[0]):
        order = np.argsort(-(img @ txt[c]))
        rel = labels[order] == c
        p_at_k.append(rel[:k].mean())
        hits = np.cumsum(rel)
        aps.append((hits[rel] / (np.flatnonzero(rel) + 1)).mean())
    return float(np.mean(p_at_k)), float(np.mean(aps))


def bench(name: str, spec: dict, items) -> dict:
    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    t0 = time.perf_counter()
    model, preprocess, tokenizer = load_candidate(name, spec)
    load_s = time.perf_counter() - t0
    labels = np.array([c for _, c in items])
    img, enc_s = encode_images(model, preprocess, items)
    deploy = encode_classes(model, tokenizer, models["text_vec"]["templates"])
    classic = encode_classes(model, tokenizer, CLIP_EUROSAT_TEMPLATES)
    p10, m_ap = retrieval(img, deploy, labels)
    res = {
        "model": name,
        "n_images": len(items),
        "top1_deploy_templates": float(((img @ deploy.T).argmax(1) == labels).mean()),
        "top1_clip_eurosat_templates": float(((img @ classic.T).argmax(1) == labels).mean()),
        "text2img_P@10": p10,
        "text2img_mAP": m_ap,
        "img_per_s": len(items) / enc_s,
        "load_s": load_s,
        "peak_rss_gb_delta": (proc.memory_info().rss - rss0) / 1e9,
    }
    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return res


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="one candidate name")
    ap.add_argument("--limit", type=int, help="class-balanced subset size (quick check)")
    a = ap.parse_args()
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    items = test_split(a.limit)
    cands = {k: v for k, v in models["text_vec_candidates"].items() if not a.only or k == a.only}
    results = []
    for name, spec in cands.items():
        try:
            r = bench(name, spec, items)
        except Exception as e:  # noqa: BLE001 - report the failure, keep benchmarking the others
            r = {"model": name, "error": f"{type(e).__name__}: {e}"}
        print(json.dumps(r), flush=True)
        results.append(r)

    REPORTS.mkdir(parents=True, exist_ok=True)
    stem = REPORTS / f"{datetime.now(UTC).date()}_bench_clip{'_limit' + str(a.limit) if a.limit else ''}"
    meta = {
        "split": "torchgeo eurosat-test.txt (RGB)",
        "n_images": len(items),
        "torch_threads": torch.get_num_threads(),
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
        "deploy_templates": models["text_vec"]["templates"],
        "classnames": CLASSNAMES,
    }
    stem.with_suffix(".json").write_text(
        json.dumps({"meta": meta, "results": results}, indent=1), encoding="utf-8"
    )
    cols = [
        "top1_deploy_templates",
        "top1_clip_eurosat_templates",
        "text2img_P@10",
        "text2img_mAP",
        "img_per_s",
    ]
    lines = ["| model | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for r in results:
        cells = (
            [f"{r[c]:.3f}" if c in r else "-" for c in cols]
            if "error" not in r
            else [r["error"][:80]] + [""] * (len(cols) - 1)
        )
        lines.append(f"| {r['model']} | " + " | ".join(cells) + " |")
    stem.with_suffix(".md").write_text(
        f"# M1 zero-shot EuroSAT ({len(items)} test images)\n\n" + "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
