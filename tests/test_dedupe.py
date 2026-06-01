"""Unit tests for deduplication utilities.

These tests intentionally import the dedupe submodule through the validation
package surface. The validation package must therefore avoid eager imports of
unrelated modules so this file remains stable under parallel collection.
"""

from astroml.validation import dedupe


def _tx(
    tx_id: str,
    payload: str = "test",
    timestamp: str = "2024-01-01",
):
    return {"id": tx_id, "payload": payload, "timestamp": timestamp}


class TestDeduplicator:
    """Tests for Deduplicator class."""

    def test_add_unique_transaction(self):
        """Should add unique transaction."""
        dedup = dedupe.Deduplicator()
        tx = _tx("1")
        result = dedup.add(tx)
        assert result is True
        # the exact hash string is based on sorting keys so we just check it was added
        assert len(dedup.seen_hashes) == 1

    def test_add_duplicate_transaction(self):
        """Should reject duplicate transaction."""
        dedup = dedupe.Deduplicator()
        tx = _tx("1")
        dedup.add(tx)
        result = dedup.add(tx)
        assert result is False

    def test_check_duplicate(self):
        """Should check for duplicates without adding."""
        dedup = dedupe.Deduplicator()
        tx = _tx("1")
        assert dedup.check(tx) is False
        dedup.add(tx)
        assert dedup.check(tx) is True

    def test_process_batch(self):
        """Should process batch and separate duplicates."""
        dedup = dedupe.Deduplicator()
        txs = [
            _tx("1", payload="test1"),
            _tx("2", payload="test2", timestamp="2024-01-02"),
            _tx("1", payload="test1"),  # duplicate
        ]
        result = dedup.process(txs)
        assert len(result.unique) == 2
        assert len(result.duplicates) == 1

    def test_filter_unique(self):
        """Should filter and return unique transactions."""
        dedup = dedupe.Deduplicator()
        txs = [
            _tx("1", payload="test1"),
            _tx("1", payload="test1"),
            _tx("2", payload="test2", timestamp="2024-01-02"),
        ]
        unique = dedup.filter_duplicates(txs, return_unique=True)
        assert len(unique) == 2

    def test_filter_duplicates_only(self):
        """Should filter and return only duplicates."""
        dedup = dedupe.Deduplicator()
        txs = [
            _tx("1", payload="test1"),
            _tx("1", payload="test1"),
            _tx("2", payload="test2", timestamp="2024-01-02"),
        ]
        duplicates = dedup.filter_duplicates(txs, return_unique=False)
        assert len(duplicates) == 1

    def test_reset(self):
        """Should clear all state."""
        dedup = dedupe.Deduplicator()
        tx = _tx("1")
        dedup.add(tx)
        dedup.reset()
        assert len(dedup.seen_hashes) == 0

    def test_conflict_tracking(self):
        """Should track conflict records."""
        dedup = dedupe.Deduplicator(track_conflicts=True)
        tx = _tx("1")
        dedup.add(tx)
        dedup.add(tx)  # duplicate
        assert len(dedup.conflicts) == 1
        assert dedup.conflicts[0].conflict_type == dedupe.ConflictType.DUPLICATE

    def test_fresh_instances_do_not_share_state(self):
        """A new Deduplicator instance must start with an empty seen set."""
        first = dedupe.Deduplicator()
        second = dedupe.Deduplicator()
        tx = _tx("shared")

        assert first.add(tx) is True
        assert second.check(tx) is False
        assert second.add(tx) is True


class TestDeduplicate:
    """Tests for deduplicate convenience function."""

    def test_deduplicate_function(self):
        """Should deduplicate transactions."""
        txs = [
            _tx("1", payload="test1"),
            _tx("2", payload="test2", timestamp="2024-01-02"),
            _tx("1", payload="test1"),
        ]
        result = dedupe.deduplicate(txs)
        assert len(result.unique) == 2
        assert len(result.duplicates) == 1
