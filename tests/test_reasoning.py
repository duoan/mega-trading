import tempfile
import unittest
from pathlib import Path

from marketfm.core.store import LocalObjectStore
from marketfm.data.ingest import FixtureIngestor
from marketfm.data.lance_store import LanceTableStore
from marketfm.reasoning import EvidenceCatalog, EvidencePackBuilder, LanceEvidenceCatalog, ReasoningRuntime, ReasoningValidationError


def _prepared_store(tmp: str) -> LocalObjectStore:
    store = LocalObjectStore(Path(tmp))
    FixtureIngestor(store).ingest()
    return store


def _prepared_lance_store(tmp: str) -> tuple[LocalObjectStore, LanceTableStore]:
    root = Path(tmp)
    store = LocalObjectStore(root / "artifacts")
    table_store = LanceTableStore(root / "lancedb")
    FixtureIngestor(store, table_store=table_store).ingest()
    return store, table_store


class ReasoningTests(unittest.TestCase):
    def test_retrieval_excludes_future_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = _prepared_store(tmp)
            catalog = EvidenceCatalog(store)

            evidence = catalog.query("ACME", "2023-01-01T00:00:00Z")

            self.assertEqual(evidence, [])

    def test_evidence_pack_includes_source_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = _prepared_store(tmp)

            pack = EvidencePackBuilder(EvidenceCatalog(store)).build("ACME", "2023-02-16T00:00:00Z", "12m")

            self.assertIn("[ev-acme-margin]", pack.prompt)
            self.assertEqual(pack.evidence_ids, ["ev-acme-margin", "ev-acme-risk"])

    def test_lance_evidence_catalog_reads_silver_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, table_store = _prepared_lance_store(tmp)
            catalog = LanceEvidenceCatalog(table_store)

            evidence = catalog.query("ACME", "2023-02-16T00:00:00Z")

            self.assertEqual([row["evidence_id"] for row in evidence], ["ev-acme-margin", "ev-acme-risk"])

    def test_runtime_returns_insufficient_evidence_when_no_sources_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = _prepared_store(tmp)

            output = ReasoningRuntime(store).reason("ACME", "2023-01-01T00:00:00Z", "12m")

            self.assertEqual(output["investment_view"], "insufficient_evidence")
            self.assertEqual(output["confidence"], 0.0)

    def test_validator_rejects_missing_citations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = _prepared_store(tmp)
            runtime = ReasoningRuntime(store)
            output = runtime.reason("ACME", "2023-02-16T00:00:00Z", "12m")
            output["evidence"] = []

            with self.assertRaises(ReasoningValidationError):
                runtime.validate(output)

    def test_runtime_writes_reasoning_output_with_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = _prepared_store(tmp)

            output = ReasoningRuntime(store, run_id="reason-test").reason("ACME", "2023-02-16T00:00:00Z", "12m")
            stored = store.read_json("runs/reason-test/reasoning-output.json")

            self.assertEqual(output["ticker"], "ACME")
            self.assertEqual(stored["lineage"]["prompt_version"], "investment-thesis-v1")
            self.assertGreater(stored["confidence"], 0.0)

    def test_runtime_can_use_lancedb_evidence_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store, table_store = _prepared_lance_store(tmp)

            output = ReasoningRuntime(store, table_store=table_store).reason("ACME", "2023-02-16T00:00:00Z", "12m")

            self.assertEqual(output["ticker"], "ACME")
            self.assertGreater(output["confidence"], 0.0)


if __name__ == "__main__":
    unittest.main()
