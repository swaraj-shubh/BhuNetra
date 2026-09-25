"""Qdrant: rebuildable vector index (DESIGN.md §2.6). PostGIS is the source of truth; write PostGIS first.

Named vectors per the two-vector rule (§2.3): text_vec (CLIP, text->image only) and eo_vec (Prithvi).
"""

import os
import uuid
from functools import lru_cache

from qdrant_client import QdrantClient, models

from configs.settings import models as model_cfg
from configs.settings import paths

COLLECTION = "tiles"
EO_VEC_DIM = int(os.environ.get("EO_VEC_DIM", "1024"))  # VERIFY Prithvi-EO-2.0-300M embed dim before M7


@lru_cache(maxsize=1)
def client() -> QdrantClient:
    url = os.environ.get("QDRANT_URL")  # set by docker/compose.yaml; embedded mode otherwise (dev laptop)
    return QdrantClient(url=url) if url else QdrantClient(path=str(paths.qdrant))


def point_id(key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def ensure_collection() -> None:
    c = client()
    if not c.collection_exists(COLLECTION):
        c.create_collection(
            COLLECTION,
            vectors_config={
                "text_vec": models.VectorParams(
                    size=model_cfg["text_vec"]["dim"], distance=models.Distance.COSINE
                ),
                "eo_vec": models.VectorParams(size=EO_VEC_DIM, distance=models.Distance.COSINE),
            },
        )


def recreate_collection() -> None:
    if client().collection_exists(COLLECTION):
        client().delete_collection(COLLECTION)
    ensure_collection()


def upsert(rows: list[tuple[str, dict, dict]]) -> None:
    """rows: (key, payload, {vector_name: vector}). Call only after the PostGIS commit succeeded."""
    client().upsert(
        COLLECTION,
        points=[
            models.PointStruct(id=point_id(key), vector={k: v.tolist() for k, v in vec.items()}, payload=p)
            for key, p, vec in rows
        ],
    )


def search(vector, using: str, k: int, query_filter: models.Filter | None = None) -> list[models.ScoredPoint]:
    if using not in ("text_vec", "eo_vec"):
        raise ValueError(using)
    return (
        client()
        .query_points(
            COLLECTION,
            query=list(map(float, vector)),
            using=using,
            query_filter=query_filter,
            limit=k,
            with_payload=True,
        )
        .points
    )
