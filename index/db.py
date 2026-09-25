"""PostGIS access (DESIGN.md §2.6: source of truth). SQLAlchemy Core + plain SQL; schema in index/schema.sql."""

import os
import secrets
import time
import uuid
from functools import lru_cache

from sqlalchemy import Engine, create_engine


@lru_cache(maxsize=1)
def engine() -> Engine:
    url = os.environ.get("DATABASE_URL")  # set by docker/compose.yaml; host tools: see docs/AWS.md
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    return create_engine(url.replace("postgresql://", "postgresql+pg8000://", 1), pool_pre_ping=True)


def uuid7() -> uuid.UUID:
    """RFC 9562 UUIDv7 (time-ordered). Python 3.11 has no uuid.uuid7."""
    ms = time.time_ns() // 1_000_000
    b = bytearray(ms.to_bytes(6, "big") + secrets.token_bytes(10))
    b[6] = 0x70 | (b[6] & 0x0F)  # version 7
    b[8] = 0x80 | (b[8] & 0x3F)  # variant 10
    return uuid.UUID(bytes=bytes(b))
