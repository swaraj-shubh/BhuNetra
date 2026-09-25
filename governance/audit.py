"""Hash-chained, append-only audit log (DESIGN.md §2.7, §7). No ML dependencies.

entry_hash = sha256(prev_hash || ts || actor || action || canonical_json(payload)), genesis prev_hash = "0"*64.
append() runs inside the caller's transaction, so the audit row commits (or rolls back) with the state change.
"""

import hashlib
import json
from datetime import UTC, datetime
from typing import Any, NamedTuple

from sqlalchemy import Connection, text

GENESIS = "0" * 64
_LOCK_KEY = 26227  # pg advisory lock id: serialises appends so the chain has no forks


class Entry(NamedTuple):
    seq: int
    ts: datetime
    actor: str
    action: str
    payload: dict[str, Any]
    prev_hash: str
    entry_hash: str


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _ts(ts: datetime) -> str:
    return ts.astimezone(UTC).isoformat(
        timespec="microseconds"
    )  # microseconds round-trip through timestamptz


def entry_hash(prev_hash: str, ts: datetime, actor: str, action: str, payload: dict[str, Any]) -> str:
    msg = prev_hash + _ts(ts) + actor + action + canonical_json(payload)
    return hashlib.sha256(msg.encode("utf-8")).hexdigest()


def verify(entries: list[Entry]) -> int | None:
    """seq of the first broken entry, or None if the whole chain verifies. Entries in seq order."""
    prev = GENESIS
    for e in entries:
        if e.prev_hash != prev or entry_hash(prev, e.ts, e.actor, e.action, e.payload) != e.entry_hash:
            return e.seq
        prev = e.entry_hash
    return None


def append(conn: Connection, actor: str, action: str, payload: dict[str, Any]) -> str:
    conn.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": _LOCK_KEY})
    prev = (
        conn.execute(text("SELECT entry_hash FROM audit_log ORDER BY seq DESC LIMIT 1")).scalar() or GENESIS
    )
    ts = datetime.now(UTC)
    h = entry_hash(prev, ts, actor, action, payload)
    conn.execute(
        text(
            "INSERT INTO audit_log (ts, actor, action, payload, prev_hash, entry_hash) "
            "VALUES (:ts, :actor, :action, CAST(:payload AS jsonb), :prev, :h)"
        ),
        {
            "ts": ts,
            "actor": actor,
            "action": action,
            "payload": canonical_json(payload),
            "prev": prev,
            "h": h,
        },
    )
    return h


def load(conn: Connection) -> list[Entry]:
    rows = conn.execute(
        text("SELECT seq, ts, actor, action, payload, prev_hash, entry_hash FROM audit_log ORDER BY seq")
    )
    return [Entry(*r) for r in rows]


if __name__ == "__main__":  # make audit-verify
    import sys

    from index import db

    with db.engine().connect() as c:
        entries = load(c)
    broken = verify(entries)
    print(json.dumps({"ok": broken is None, "entries": len(entries), "broken_at": broken}))
    sys.exit(1 if broken else 0)
