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
PAYLOAD_INDEXES = {
    "aoi": models.PayloadSchemaType.KEYWORD,
    "tile_id": models.PayloadSchemaType.KEYWORD,  # M4: score event tiles (MatchAny)
    "season": models.PayloadSchemaType.KEYWORD,
    "start_date": models.PayloadSchemaType.DATETIME,
    "end_date": models.PayloadSchemaType.DATETIME,
}
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
    # filtered search scans every payload without these (dholera: 4.7 s vs 0.36 s unfiltered)
    for field, schema in PAYLOAD_INDEXES.items():
        c.create_payload_index(COLLECTION, field, schema)


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


def count(match: dict[str, str]) -> int:
    if not client().collection_exists(COLLECTION):
        return 0
    flt = models.Filter(
        must=[models.FieldCondition(key=k, match=models.MatchValue(value=v)) for k, v in match.items()]
    )
    return client().count(COLLECTION, count_filter=flt, exact=True).count


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


def search_best_per_tile(
    vector, using: str, n_tiles: int, query_filter: models.Filter | None = None
) -> list[models.ScoredPoint]:
    """Best-scoring point per tile_id (one grouped query; needs the tile_id payload index)."""
    if using not in ("text_vec", "eo_vec"):
        raise ValueError(using)
    groups = (
        client()
        .query_points_groups(
            COLLECTION,
            query=list(map(float, vector)),
            using=using,
            group_by="tile_id",
            group_size=1,
            limit=n_tiles,
            query_filter=query_filter,
            with_payload=True,
        )
        .groups
    )
    return [g.hits[0] for g in groups if g.hits]
