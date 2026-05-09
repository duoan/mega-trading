import unittest

from mega_trading.core.hashing import stable_hash
from mega_trading.core.schemas import (
    DocumentRecord,
    Manifest,
    MarketDataRecord,
    SchemaValidationError,
)


class SchemaTests(unittest.TestCase):
    def test_model_visible_records_require_as_of_time(self) -> None:
        with self.assertRaises(SchemaValidationError):
            DocumentRecord(
                document_id="doc-1",
                entity_id="entity-1",
                ticker="ACME",
                source_type="10-K",
                title="ACME 10-K",
                text="Management discussion",
                source_uri="fixture://doc-1",
                published_at="2023-02-01T00:00:00Z",
                accepted_at="2023-02-01T00:00:00Z",
                as_of_time="",
                source_ids=["raw-1"],
            )

    def test_invalid_timestamp_fails(self) -> None:
        with self.assertRaises(SchemaValidationError):
            DocumentRecord(
                document_id="doc-1",
                entity_id="entity-1",
                ticker="ACME",
                source_type="10-K",
                title="ACME 10-K",
                published_at="not-a-date",
                accepted_at="2023-02-01T00:00:00Z",
                as_of_time="2023-02-01T00:00:00Z",
                text="Revenue increased.",
                source_uri="fixture://doc-1",
                source_ids=["raw-1"],
            )

    def test_market_data_record_allows_date_without_time(self) -> None:
        record = MarketDataRecord(
            market_data_id="px-1",
            ticker="ACME",
            date="2023-01-03",
            adjusted_close=100.0,
            provider="fixture",
            source_ids=["raw-market_data-1"],
        )

        self.assertEqual(record.date, "2023-01-03")

    def test_manifest_hash_is_deterministic(self) -> None:
        left = Manifest(
            manifest_id="manifest-1",
            artifact_type="samples",
            paths=["stage=04_corpus/mixture=demo/samples.jsonl"],
            metadata={"b": 2, "a": 1},
        )
        right = Manifest(
            manifest_id="manifest-1",
            artifact_type="samples",
            paths=["stage=04_corpus/mixture=demo/samples.jsonl"],
            metadata={"a": 1, "b": 2},
        )

        self.assertEqual(left.content_hash(), right.content_hash())
        self.assertEqual(left.content_hash(), stable_hash(right.to_dict()))

if __name__ == "__main__":
    unittest.main()
