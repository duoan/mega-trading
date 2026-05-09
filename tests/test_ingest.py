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
            self.assertEqual(manifest.paths, ["stage=01_raw/source=fixture/order_flow.jsonl"])
            self.assertEqual(manifest.metadata["source"], "fixture")

    def test_fixture_ingestor_implements_ingestor_interface(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))

            ingestor: Ingestor[FixtureIngestRequest] = FixtureIngestor(store)
            result = ingestor.ingest(FixtureIngestRequest())

            self.assertEqual(result.normalized_counts["order_flow"], 24)

    def test_fixture_ingest_writes_normalized_order_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            result = FixtureIngestor(store).ingest()

            events = store.read_jsonl("stage=02_normalized/family=order_flow/source=fixture.jsonl")

            self.assertEqual(len(events), 24)
            self.assertEqual(events[0]["ticker"], "ACME")
            self.assertEqual(result.normalized_counts["order_flow"], 24)

    def test_bad_records_go_to_quarantine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            FixtureIngestor(store).ingest()

            quarantined = store.read_jsonl("quarantine/fixture/bad_records.jsonl")

            self.assertEqual(len(quarantined), 1)
            self.assertEqual(quarantined[0]["reason"], "invalid_event")


if __name__ == "__main__":
    unittest.main()
