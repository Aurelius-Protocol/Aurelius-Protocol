"""Local observations are durable, idempotent and never infer a charge from serving."""

import pytest

from subnet.miner.journal import AttemptJournal, identity_for

pytestmark = pytest.mark.miner


def test_restart_keeps_independent_source_events(tmp_path):
    path = tmp_path / "journal.sqlite3"
    identity = identity_for(
        {"task_type": "synthetic", "schema_version": 1, "config": {"payload": "a"}},
        "miner",
        window=4,
    )
    journal = AttemptJournal(path)
    for _ in range(2):
        journal.record(identity, "envelope_created", "local", {"validator_hotkey": "validator"})
    rows = AttemptJournal(path).attempts()
    assert len(rows) == 1
    assert [e["stage"] for e in rows[0]["events"]] == ["envelope_created"]
    assert rows[0]["identity"]["content_hash"] == identity["content_hash"]
    assert "signature" not in str(rows)
