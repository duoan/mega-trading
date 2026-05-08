import unittest

from marketfm.core.config import DataMixtureConfig, MixtureSource
from marketfm.core.hashing import stable_hash
from marketfm.core.schemas import (
    CorpusRecord,
    DocumentRecord,
    EvidenceRecord,
    Manifest,
    PriceRecord,
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
            EvidenceRecord(
                evidence_id="ev-1",
                entity_id="entity-1",
                ticker="ACME",
                source_type="10-K",
                document_id="doc-1",
                timestamp="not-a-date",
                as_of_time="2023-02-01T00:00:00Z",
                text="Revenue increased.",
                uri="fixture://doc-1",
            )

    def test_corpus_record_preserves_lineage(self) -> None:
        record = CorpusRecord(
            corpus_id="corpus-1",
            task_type="cpt_text",
            mixture_name="demo",
            entity_id="entity-1",
            ticker="ACME",
            as_of_time="2023-02-01T00:00:00Z",
            text="ACME revenue increased.",
            source_ids=["doc-1"],
            evidence_ids=["ev-1"],
            quality_score=0.9,
        )

        self.assertEqual(record.source_ids, ["doc-1"])
        self.assertEqual(record.evidence_ids, ["ev-1"])

    def test_price_record_allows_date_without_time(self) -> None:
        record = PriceRecord(
            price_id="px-1",
            ticker="ACME",
            date="2023-01-03",
            adjusted_close=100.0,
            provider="fixture",
            source_ids=["raw-price-1"],
        )

        self.assertEqual(record.date, "2023-01-03")

    def test_manifest_hash_is_deterministic(self) -> None:
        left = Manifest(
            manifest_id="manifest-1",
            artifact_type="corpus",
            paths=["corpus/demo.jsonl"],
            metadata={"b": 2, "a": 1},
        )
        right = Manifest(
            manifest_id="manifest-1",
            artifact_type="corpus",
            paths=["corpus/demo.jsonl"],
            metadata={"a": 1, "b": 2},
        )

        self.assertEqual(left.content_hash(), right.content_hash())
        self.assertEqual(left.content_hash(), stable_hash(right.to_dict()))

    def test_data_mixture_config_requires_positive_weights(self) -> None:
        with self.assertRaises(SchemaValidationError):
            DataMixtureConfig(
                name="bad",
                sources=[MixtureSource(name="filings", weight=0.0, task_type="cpt_text")],
            )


if __name__ == "__main__":
    unittest.main()
