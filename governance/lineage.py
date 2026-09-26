"""Lineage records for derived artifacts (DESIGN.md §2.7). No ML dependencies (§5 module boundaries)."""

import json
import subprocess
from datetime import UTC, datetime
from typing import Any

from configs.settings import PROCESSING_VERSION, paths


def git_sha() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=paths.root, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, cwd=paths.root, check=False
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "nogit"


def now_ts() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def record(
    kind: str, inputs: dict[str, Any], params: dict[str, Any], models: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "kind": kind,
        "inputs": inputs,
        "params": params,
        "models": models or {},
        "processing_version": PROCESSING_VERSION,
        "code": git_sha(),
        "created_ts": now_ts(),
    }


def append(log_name: str, rec: dict[str, Any]) -> None:
    """Append-only JSONL under data/manifests/ (until the PostGIS audit_log exists)."""
    paths.logs.mkdir(parents=True, exist_ok=True)
    with open(paths.logs / f"{log_name}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
