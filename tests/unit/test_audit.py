from datetime import UTC, datetime

from governance import audit


def _chain(n):
    rows, prev = [], audit.GENESIS
    for i in range(n):
        ts = datetime(2026, 9, 26, 12, 0, i, 123456, tzinfo=UTC)
        payload = {"scene_id": f"S{i}", "n": i, "note": "ünïcode"}
        h = audit.entry_hash(prev, ts, "worker", "ingest.scene", payload)
        rows.append(audit.Entry(i + 1, ts, "worker", "ingest.scene", payload, prev, h))
        prev = h
    return rows


def test_chain_verifies():
    assert audit.verify(_chain(5)) is None
    assert audit.verify([]) is None


def test_hash_is_order_independent_for_payload_keys():
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    assert audit.entry_hash(audit.GENESIS, ts, "a", "x", {"a": 1, "b": 2}) == audit.entry_hash(
        audit.GENESIS, ts, "a", "x", {"b": 2, "a": 1}
    )


def test_editing_any_field_breaks_at_that_seq():
    for field in ("payload", "actor", "action", "ts"):
        rows = _chain(5)
        e = rows[2]
        changed = {
            "payload": {**e.payload, "n": 99},
            "actor": "mallory",
            "action": "ingest.delete",
            "ts": e.ts.replace(second=59),
        }[field]
        rows[2] = e._replace(**{field: changed})
        assert audit.verify(rows) == 3, field


def test_deleting_a_row_breaks_the_chain():
    rows = _chain(5)
    del rows[1]
    assert audit.verify(rows) == 3  # seq 3's prev_hash no longer matches seq 1
