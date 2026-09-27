"""eo_vec: Prithvi-EO-2.0-300M-TL encoder for image->image search, clustering, change evidence (DESIGN.md §2.3, §7).

Facts from models/prithvi-eo-2.0-300m-tl/config.json (model card): img 224, patch 16, embed 1024, 6 bands
(HLS B02 B03 B04 B05 B06 B07 = our blue green red nir_narrow swir1 swir2), mean/std in reflectance x 10000,
time + location coordinate encodings. Local weights only, sha256-verified (§2.8, M8).

Encoder input rule (DESIGN.md §7): run on 224 px (2240 m) windows of a seasonal composite and average the patch tokens
inside each 640 m tile (4 x 4 patches at 10 m). One forward pass per window, not per tile.
"""

import importlib.util
import json
import math
import sys
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from configs.settings import paths

NAME = "prithvi-eo-2.0-300m-tl"
DIR = paths.models / NAME
WEIGHTS_FILE = "Prithvi_EO_V2_300M_TL.pt"
BANDS = ("blue", "green", "red", "nir_narrow", "swir1", "swir2")  # composite band names, model order
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def windows(h: int, w: int, size: int) -> list[tuple[int, int]]:
    """Top-left (row, col) of non-overlapping windows covering an h x w raster (last ones padded)."""
    return [
        (r, c)
        for r in range(0, math.ceil(h / size) * size, size)
        for c in range(0, math.ceil(w / size) * size, size)
    ]


def normalise(x: np.ndarray, mean, std) -> np.ndarray:
    """(6, H, W) reflectance 0-1 -> model units; NaN (no valid pixel) -> 0 (= the band mean)."""
    m = np.asarray(mean, np.float32)[:, None, None]
    s = np.asarray(std, np.float32)[:, None, None]
    out = (x * 10000.0 - m) / s
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


def pool_tiles(
    fmap: np.ndarray, offsets: list[tuple[int, int]], tile_px: int, patch_px: int = 16
) -> np.ndarray:
    """fmap (Hp, Wp, D) patch tokens; offsets = tiles' top-left (row_px, col_px). L2-normalised mean per tile."""
    size = tile_px
    n = size // patch_px
    out = np.empty((len(offsets), fmap.shape[2]), np.float32)
    for i, (r, c) in enumerate(offsets):
        if r % patch_px or c % patch_px:
            raise ValueError(f"tile at px ({r}, {c}) is not aligned to {patch_px} px patches")
        v = (
            fmap[r // patch_px : r // patch_px + n, c // patch_px : c // patch_px + n]
            .reshape(-1, fmap.shape[2])
            .mean(0)
        )
        out[i] = v / (np.linalg.norm(v) or 1.0)
    return out


@lru_cache(maxsize=1)
def _load():
    from governance.model_registry import verified

    cfg = json.loads((DIR / "config.json").read_text(encoding="utf-8"))["pretrained_cfg"]
    spec = importlib.util.spec_from_file_location("prithvi_mae_local", DIR / "prithvi_mae.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["prithvi_mae_local"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    enc = mod.PrithviViT(
        img_size=cfg["img_size"],
        patch_size=cfg["patch_size"],
        num_frames=1,
        in_chans=cfg["in_chans"],
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        mlp_ratio=cfg["mlp_ratio"],
        coords_encoding=cfg["coords_encoding"],
        coords_scale_learn=cfg["coords_scale_learn"],
    )
    sd = torch.load(verified(NAME, WEIGHTS_FILE), map_location="cpu", weights_only=True)
    sd = {k[len("encoder.") :]: v for k, v in sd.items() if k.startswith("encoder.") and "pos_embed" not in k}
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    if unexpected or [m for m in missing if "pos_embed" not in m]:
        raise RuntimeError(f"Prithvi encoder load mismatch: missing={missing} unexpected={unexpected}")
    enc.to(DEVICE).eval()
    if DEVICE == "cuda":
        enc.half()
    return enc, cfg


@lru_cache(maxsize=1)
def version() -> str:
    from governance.model_registry import verified
    from ingest.validate import sha256

    return f"{NAME}@{sha256(verified(NAME, WEIGHTS_FILE))[:12]}"


@torch.no_grad()
def feature_map(x: np.ndarray, mid_date: str, lat: float, lon: float, batch: int = 8) -> np.ndarray:
    """(6, H, W) composite reflectance -> (ceil(H/16), ceil(W/16), 1024) last-layer patch tokens."""
    enc, cfg = _load()
    size, p = cfg["img_size"], cfg["patch_size"][-1]
    h, w = x.shape[1:]
    xn = normalise(x, cfg["mean"], cfg["std"])
    hp, wp = math.ceil(h / size) * size, math.ceil(w / size) * size
    xn = np.pad(xn, ((0, 0), (0, hp - h), (0, wp - w)))
    d = date.fromisoformat(mid_date)
    tc = torch.tensor([[[d.year, d.timetuple().tm_yday]]], dtype=torch.float32)
    lc = torch.tensor([[lat, lon]], dtype=torch.float32)
    fmap = np.zeros((hp // p, wp // p, cfg["embed_dim"]), np.float32)
    ws = windows(h, w, size)
    dt = torch.float16 if DEVICE == "cuda" else torch.float32
    for i in range(0, len(ws), batch):
        chunk = ws[i : i + batch]
        xb = torch.from_numpy(np.stack([xn[:, r : r + size, c : c + size] for r, c in chunk])).to(DEVICE, dt)
        n = len(chunk)
        tok = enc.forward_features(xb, tc.repeat(n, 1, 1).to(DEVICE, dt), lc.repeat(n, 1).to(DEVICE, dt))[-1]
        tok = tok[:, 1:, :].float().cpu().numpy().reshape(n, size // p, size // p, -1)
        for (r, c), t in zip(chunk, tok, strict=True):
            fmap[r // p : (r + size) // p, c // p : (c + size) // p] = t
    return fmap[: math.ceil(h / p), : math.ceil(w / p)]


def embed_composite(
    tif: Path, tiles_px: list[tuple[int, int]], tile_len_px: int, mid_date: str, lat: float, lon: float
) -> np.ndarray:
    import rasterio

    with rasterio.open(tif) as ds:
        x = np.stack([ds.read(ds.descriptions.index(b) + 1) for b in BANDS]).astype(np.float32)
    return pool_tiles(feature_map(x, mid_date, lat, lon), tiles_px, tile_px=tile_len_px)
