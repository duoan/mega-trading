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

            store.write_table("stage_02_normalized_entities_fixture", [{"entity_id": "one", "ticker": "ONE"}])

            self.assertEqual(store.read_rows("stage_02_normalized_entities_fixture"), [{"entity_id": "one", "ticker": "ONE"}])

    def test_write_table_overwrites_existing_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LanceTableStore(Path(tmp))

            store.write_table("stage_02_normalized_entities_fixture", [{"entity_id": "one", "ticker": "ONE"}])
            store.write_table("stage_02_normalized_entities_fixture", [{"entity_id": "two", "ticker": "TWO"}])

            self.assertEqual(store.read_rows("stage_02_normalized_entities_fixture"), [{"entity_id": "two", "ticker": "TWO"}])

    def test_lance_table_names_are_stable(self) -> None:
        tables = LanceTables()

        self.assertEqual(tables.normalized("fundamentals", "sec"), "stage_02_normalized_fundamentals_sec")
        self.assertEqual(tables.corpus("demo", "cpt"), "stage_04_corpus_demo_cpt")

    def test_fixture_ingestor_writes_normalized_lance_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            object_store = LocalObjectStore(root / "artifacts")
            table_store = LanceTableStore(root / "lancedb")

            FixtureIngestor(object_store, table_store=table_store).ingest()

            entities = table_store.read_rows("stage_02_normalized_entities_fixture")
            evidence = table_store.read_rows("stage_02_normalized_evidence_fixture")

            self.assertEqual(len(entities), 2)
            self.assertEqual(evidence[0]["ticker"], "ACME")


if __name__ == "__main__":
    unittest.main()
