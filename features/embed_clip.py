"""text_vec: remote-sensing CLIP for text->image search ONLY (DESIGN.md §2.3). Local weights only (§2.8)."""

from functools import lru_cache

import numpy as np
import open_clip
import torch
from PIL import Image

from configs.settings import models, paths
from ingest.validate import sha256

CFG = models["text_vec"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS = paths.models / CFG["name"] / CFG["file"]


@lru_cache(maxsize=1)
def _load():
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"{WEIGHTS} missing; run: python scripts/stage_data.py models")
    model, _, preprocess = open_clip.create_model_and_transforms(CFG["arch"])
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    model.to(DEVICE).eval()
    return model, preprocess, open_clip.get_tokenizer(CFG["arch"])


@lru_cache(maxsize=1)
def version() -> str:
    return f"{CFG['name']}@{sha256(WEIGHTS)[:12]}"


def _norm(x: torch.Tensor) -> np.ndarray:
    return torch.nn.functional.normalize(x, dim=-1).cpu().numpy().astype(np.float32)


@torch.no_grad()
def embed_images(images: list[np.ndarray | Image.Image], batch: int = 32) -> np.ndarray:
    """RGB chips (any size; the model's preprocess resizes). Returns L2-normalised (n, dim)."""
    model, preprocess, _ = _load()
    out = []
    for i in range(0, len(images), batch):
        pil = [im if isinstance(im, Image.Image) else Image.fromarray(im) for im in images[i : i + batch]]
        out.append(
            _norm(model.encode_image(torch.stack([preprocess(p.convert("RGB")) for p in pil]).to(DEVICE)))
        )
    return np.concatenate(out)


@torch.no_grad()
def embed_text(query: str) -> np.ndarray:
    model, _, tokenizer = _load()
    e = _norm(model.encode_text(tokenizer([t.format(query) for t in CFG["templates"]]).to(DEVICE)))
    mean = e.mean(axis=0)
    return mean / np.linalg.norm(mean)
