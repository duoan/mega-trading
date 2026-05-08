import tempfile
import unittest
from pathlib import Path

from marketfm.data.ingest import FixtureIngestor
from marketfm.data.lance_store import LanceTableStore, LanceTables
from marketfm.core.store import LocalObjectStore


class LanceStoreTests(unittest.TestCase):
    def test_lance_table_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LanceTableStore(Path(tmp))

            store.write_table("silver_entities_fixture", [{"entity_id": "one", "ticker": "ONE"}])

            self.assertEqual(store.read_rows("silver_entities_fixture"), [{"entity_id": "one", "ticker": "ONE"}])

    def test_lance_table_names_are_stable(self) -> None:
        tables = LanceTables()

        self.assertEqual(tables.silver("fundamentals", "sec"), "silver_fundamentals_sec")
        self.assertEqual(tables.corpus("demo", "cpt"), "corpus_demo_cpt")

    def test_fixture_ingestor_writes_silver_lance_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            object_store = LocalObjectStore(root / "artifacts")
            table_store = LanceTableStore(root / "lancedb")

            FixtureIngestor(object_store, table_store=table_store).ingest()

            entities = table_store.read_rows("silver_entities_fixture")
            evidence = table_store.read_rows("silver_evidence_fixture")

            self.assertEqual(len(entities), 2)
            self.assertEqual(evidence[0]["ticker"], "ACME")


if __name__ == "__main__":
    unittest.main()
