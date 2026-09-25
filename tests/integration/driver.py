"""Integration driver (DESIGN.md §11): run by test_pipeline.py in a subprocess with SIH_ROOT=<tmp root>.

Synthetic AOI -> worker ingest -> /search/text, with the network blocked (only loopback may connect) and CLIP replaced
by a deterministic embedder (CI has no weights; this tests the plumbing, not the model). Prints one JSON result.
"""

import json
import socket
import sys

import numpy as np

_connect = socket.socket.connect


def _loopback_only(self, address):  # offline proof: anything but loopback / unix sockets is refused
    host = address[0] if isinstance(address, tuple) else "unix"
    if host not in ("unix", "127.0.0.1", "::1", "localhost"):
        raise OSError(f"network disabled in integration test: {address}")
    return _connect(self, address)


socket.socket.connect = _loopback_only

from features import embed_clip

DIM = 512


def _fake_images(images, batch: int = 32):
    out = np.zeros((len(images), DIM), np.float32)
    for i, im in enumerate(images):
        b = float(np.asarray(im, np.float32).mean() / 255.0)
        out[i, :2] = (b, 1 - b)  # bright tiles point to e0, dark ones to e1
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def _fake_text(query: str):
    v = np.zeros(DIM, np.float32)
    v[0 if "bright" in query else 1] = 1.0
    return v


embed_clip.embed_images = _fake_images
embed_clip.embed_text = _fake_text
embed_clip.version = lambda: "fake-clip@0"

from fastapi.testclient import TestClient

from api.main import app
from governance import audit
from index import db, reconcile
from worker import ingest


def search(c: TestClient, q: str) -> list[dict]:
    r = c.post("/search/text", json={"query": q, "aoi": "synth", "k": 5})
    r.raise_for_status()
    return r.json()["sites"]


def main() -> None:
    first = ingest.ingest_aoi("synth")
    again = ingest.ingest_aoi("synth")
    c = TestClient(app)
    hits = search(c, "bright structure")
    with db.engine().connect() as conn:
        broken = audit.verify(audit.load(conn))
        n_audit = len(audit.load(conn))
    reconcile.rebuild()
    after = search(c, "bright structure")
    chip = c.get(hits[0]["chip_url"]) if hits else None
    print(
        json.dumps(
            {
                "first": first,
                "again": again,
                "top": hits[0] if hits else None,
                "ranking": [h["tile_id"] for h in hits],
                "ranking_after_reconcile": [h["tile_id"] for h in after],
                "audit_broken_at": broken,
                "audit_entries": n_audit,
                "chip_status": chip.status_code if chip else None,
            }
        )
    )


if __name__ == "__main__":
    sys.exit(main())
