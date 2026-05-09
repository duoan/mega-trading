import tempfile
import unittest
from pathlib import Path

from mega_trading.data.ingest import FixtureIngestor
from mega_trading.data.lance_store import LanceTableStore, LanceTables
from mega_trading.core.store import LocalObjectStore


class LanceStoreTests(unittest.TestCase):
    def test_lance_table_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LanceTableStore(Path(tmp))

            store.write_table("stage_02_normalized_order_flow_fixture", [{"event_id": "one", "ticker": "ONE"}])

            self.assertEqual(store.read_rows("stage_02_normalized_order_flow_fixture"), [{"event_id": "one", "ticker": "ONE"}])

    def test_write_table_overwrites_existing_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LanceTableStore(Path(tmp))

            store.write_table("stage_02_normalized_order_flow_fixture", [{"event_id": "one", "ticker": "ONE"}])
            store.write_table("stage_02_normalized_order_flow_fixture", [{"event_id": "two", "ticker": "TWO"}])

            self.assertEqual(store.read_rows("stage_02_normalized_order_flow_fixture"), [{"event_id": "two", "ticker": "TWO"}])

    def test_lance_table_names_are_stable(self) -> None:
        tables = LanceTables()

        self.assertEqual(tables.normalized("order_flow", "fixture"), "stage_02_normalized_order_flow_fixture")
        self.assertEqual(tables.corpus("demo", "samples"), "stage_04_corpus_demo_samples")

    def test_fixture_ingestor_writes_normalized_lance_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            object_store = LocalObjectStore(root / "artifacts")
            table_store = LanceTableStore(root / "lancedb")

            FixtureIngestor(object_store, table_store=table_store).ingest()

            events = table_store.read_rows("stage_02_normalized_order_flow_fixture")

            self.assertEqual(len(events), 24)
            self.assertEqual(events[0]["ticker"], "ACME")


if __name__ == "__main__":
    unittest.main()
