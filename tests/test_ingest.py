import tempfile
import unittest
from pathlib import Path

from marketfm.data.ingest import FixtureIngestor
from marketfm.core.store import LocalObjectStore


class IngestTests(unittest.TestCase):
    def test_fixture_ingest_writes_bronze_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            result = FixtureIngestor(store).ingest()

            manifest = store.read_manifest(result.bronze_manifest_path)

            self.assertEqual(manifest.artifact_type, "bronze")
            self.assertIn("bronze/fixture/entities.jsonl", manifest.paths)
            self.assertEqual(manifest.metadata["source"], "fixture")

    def test_fixture_ingest_writes_normalized_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            result = FixtureIngestor(store).ingest()

            entities = store.read_jsonl("silver/entities/fixture.jsonl")
            documents = store.read_jsonl("silver/documents/fixture.jsonl")
            fundamentals = store.read_jsonl("silver/fundamentals/fixture.jsonl")

            self.assertEqual(len(entities), 2)
            self.assertEqual(documents[0]["ticker"], "ACME")
            self.assertEqual(fundamentals[0]["concept"], "Revenue")
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
