import tempfile
import unittest
from pathlib import Path

from marketfm.core.config import DataMixtureConfig, MixtureSource
from marketfm.data.corpus import CorpusBuilder
from marketfm.data.ingest import FixtureIngestor
from marketfm.core.store import LocalObjectStore
from marketfm.data.tokenize import MalformedCorpusError, ShardBuilder, SimpleTokenizer, StreamShardBuilder


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
            store.write_jsonl("stage=04_corpus/mixture=demo/cpt.jsonl", [{"corpus_id": "bad"}])

            with self.assertRaises(MalformedCorpusError):
                ShardBuilder(store, sequence_length=8).build("demo", "cpt")

    def test_stream_shard_builder_writes_compact_model_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            store.write_jsonl("stage=04_corpus/mixture=public/samples.jsonl", [_sample()])

            result = StreamShardBuilder(store).build("public")
            rows = store.read_jsonl(result.shard_path)
            manifest = store.read_manifest(result.manifest_path)

            self.assertEqual(result.num_samples, 1)
            self.assertEqual(rows[0]["sample_id"], "sample-ACME-2024-01-02")
            self.assertEqual(rows[0]["return_label"], "outperform")
            self.assertEqual(rows[0]["risk_label"], "low")
            self.assertEqual(rows[0]["price_returns"], [0.0, 0.1])
            self.assertEqual(rows[0]["fundamental_values"], [100.0])
            self.assertEqual(manifest.metadata["artifact"], "multi_stream_samples")


def _sample() -> dict[str, object]:
    return {
        "sample_id": "sample-ACME-2024-01-02",
        "ticker": "ACME",
        "as_of_time": "2024-01-02T00:00:00Z",
        "price_window": [
            {"date": "2024-01-01", "adjusted_close": 100.0, "price_id": "px-1", "source_ids": ["px-1"]},
            {"date": "2024-01-02", "adjusted_close": 110.0, "price_id": "px-2", "source_ids": ["px-2"]},
        ],
        "fundamental_facts": [
            {
                "fundamental_id": "fact-1",
                "concept": "Revenue",
                "value": 100.0,
                "as_of_time": "2023-12-31T00:00:00Z",
                "source_ids": ["fact-1"],
            }
        ],
        "text_evidence": [{"evidence_id": "ev-1", "text": "Revenue improved.", "source_ids": ["ev-1"]}],
        "labels": {"forward_return_bucket": "outperform", "risk_bucket": "low", "forward_return": 0.08},
        "source_ids": ["px-1", "px-2", "fact-1", "ev-1"],
        "evidence_ids": ["ev-1"],
    }


if __name__ == "__main__":
    unittest.main()
