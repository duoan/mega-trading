import tempfile
import unittest
from pathlib import Path

from marketfm.config import DataMixtureConfig, MixtureSource
from marketfm.corpus import CorpusBuilder
from marketfm.ingest import FixtureIngestor
from marketfm.store import LocalObjectStore
from marketfm.tokenize import MalformedCorpusError, ShardBuilder, SimpleTokenizer


class TokenizeTests(unittest.TestCase):
    def test_simple_tokenizer_is_deterministic(self) -> None:
        tokenizer = SimpleTokenizer.fit(["ACME revenue increased.", "ACME margin improved."])

        self.assertEqual(tokenizer.encode("ACME revenue increased."), tokenizer.encode("ACME revenue increased."))
        self.assertEqual(tokenizer.decode(tokenizer.encode("ACME margin improved.")), "acme margin improved.")

    def test_shard_builder_writes_manifest_with_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            FixtureIngestor(store).ingest()
            CorpusBuilder(store).build(DataMixtureConfig("demo", [MixtureSource("filings", 1.0, "cpt_text")]))

            result = ShardBuilder(store, sequence_length=8).build("demo", "cpt")
            rows = store.read_jsonl(result.shard_path)
            manifest = store.read_manifest(result.manifest_path)

            self.assertGreater(len(rows), 0)
            self.assertEqual(manifest.metadata["tokenizer"], "simple-v1")
            self.assertEqual(manifest.metadata["sequence_length"], "8")
            self.assertIn("cpt-doc-acme-2022-10k", rows[0]["source_corpus_ids"])

    def test_sequence_packing_tracks_efficiency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            FixtureIngestor(store).ingest()
            CorpusBuilder(store).build(DataMixtureConfig("demo", [MixtureSource("filings", 1.0, "cpt_text")]))

            result = ShardBuilder(store, sequence_length=16).build("demo", "cpt")

            self.assertGreater(result.packing_efficiency, 0.0)
            self.assertLessEqual(result.packing_efficiency, 1.0)

    def test_malformed_corpus_record_fails_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            store.write_jsonl("corpus/demo/cpt.jsonl", [{"corpus_id": "bad"}])

            with self.assertRaises(MalformedCorpusError):
                ShardBuilder(store, sequence_length=8).build("demo", "cpt")


if __name__ == "__main__":
    unittest.main()
