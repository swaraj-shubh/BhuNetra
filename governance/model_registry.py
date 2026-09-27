"""Model weight verification (DESIGN.md §9, M8): a model file is used only if its sha256 matches manifests/models.yaml."""

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from configs.settings import paths


class WeightMismatch(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(manifest_path: Path | None = None) -> dict[str, Any]:
    return (
        yaml.safe_load((manifest_path or paths.manifests / "models.yaml").read_text(encoding="utf-8")) or {}
    )


@lru_cache(maxsize=32)
def verified(
    name: str, rel_file: str, manifest_path: Path | None = None, models_dir: Path | None = None
) -> Path:
    """Path to models/<name>/<rel_file> after checking its sha256; raises WeightMismatch otherwise."""
    entry = manifest(manifest_path).get(name)
    expected = (entry or {}).get("sha256", {}).get(rel_file)
    f = (models_dir or paths.models) / name / rel_file
    if expected is None:
        raise WeightMismatch(f"{name}/{rel_file} is not in manifests/models.yaml")
    if not f.exists():
        raise WeightMismatch(f"{f} missing")
    got = _sha256(f)
    if got != expected:
        raise WeightMismatch(f"{name}/{rel_file}: sha256 {got[:12]} != manifest {expected[:12]}")
    return f


def verify_all(manifest_path: Path | None = None, models_dir: Path | None = None) -> dict[str, str]:
    out = {}
    for name, e in manifest(manifest_path).items():
        for rel in e.get("sha256", {}):
            try:
                verified.__wrapped__(name, rel, manifest_path, models_dir)
                out[f"{name}/{rel}"] = "ok"
            except WeightMismatch as ex:
                out[f"{name}/{rel}"] = str(ex)
    return out
