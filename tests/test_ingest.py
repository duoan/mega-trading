import tempfile
import unittest
from pathlib import Path

from mega_trading.core.store import LocalObjectStore
from mega_trading.data.ingest import FixtureIngestRequest, FixtureIngestor, Ingestor


class IngestTests(unittest.TestCase):
    def test_fixture_ingest_writes_raw_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            result = FixtureIngestor(store).ingest(FixtureIngestRequest())

            manifest = store.read_manifest(result.raw_manifest_path)

            self.assertEqual(manifest.artifact_type, "raw")
            self.assertIn("stage=01_raw/source=fixture/entities.jsonl", manifest.paths)
            self.assertEqual(manifest.metadata["source"], "fixture")

    def test_fixture_ingestor_implements_ingestor_interface(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))

            ingestor: Ingestor[FixtureIngestRequest] = FixtureIngestor(store)
            result = ingestor.ingest(FixtureIngestRequest())

            self.assertEqual(result.normalized_counts["entities"], 2)

    def test_fixture_ingest_writes_normalized_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            result = FixtureIngestor(store).ingest()

            entities = store.read_jsonl("stage=02_normalized/family=entities/source=fixture.jsonl")
            documents = store.read_jsonl("stage=02_normalized/family=documents/source=fixture.jsonl")
            sec_filings = store.read_jsonl("stage=02_normalized/family=sec_filings/source=fixture.jsonl")

            self.assertEqual(len(entities), 2)
            self.assertEqual(documents[0]["ticker"], "ACME")
            self.assertEqual(sec_filings[0]["concept"], "Revenue")
            self.assertEqual(result.normalized_counts["documents"], 2)

    def test_bad_records_go_to_quarantine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            FixtureIngestor(store).ingest()

            quarantined = store.read_jsonl("quarantine/fixture/bad_records.jsonl")

            self.assertEqual(len(quarantined), 3)
            self.assertEqual({row["reason"] for row in quarantined}, {"stale_feed", "future_leakage", "missing_entity"})

    def test_duplicate_records_are_counted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            result = FixtureIngestor(store).ingest()

            self.assertEqual(result.quality_summary["duplicate_records"], 0)


if __name__ == "__main__":
    unittest.main()
