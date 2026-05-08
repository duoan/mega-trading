import tempfile
import unittest
from pathlib import Path

from mega_trading.core.config import DataMixtureConfig, MixtureSource
from mega_trading.data.corpus import CorpusBuilder, LeakageError
from mega_trading.data.ingest import FixtureIngestor
from mega_trading.core.store import LocalObjectStore


class CorpusBuilderTests(unittest.TestCase):
    def test_builds_cpt_sft_and_preference_corpora(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            FixtureIngestor(store).ingest()
            mixture = DataMixtureConfig(
                name="demo",
                sources=[
                    MixtureSource("filings", 0.5, "cpt_text"),
                    MixtureSource("qa", 0.3, "sft_instruction"),
                    MixtureSource("preference", 0.2, "preference_pair"),
                ],
            )

            result = CorpusBuilder(store).build(mixture)

            cpt = store.read_jsonl("stage=04_corpus/mixture=demo/cpt.jsonl")
            sft = store.read_jsonl("stage=04_corpus/mixture=demo/sft.jsonl")
            preference = store.read_jsonl("stage=04_corpus/mixture=demo/preference.jsonl")
            manifest = store.read_manifest(result.manifest_path)

            self.assertEqual(len(cpt), 2)
            self.assertEqual(len(sft), 2)
            self.assertEqual(len(preference), 4)
            self.assertEqual(manifest.metadata["mixture_name"], "demo")
            self.assertEqual(result.counts["preference"], 4)

    def test_corpus_records_preserve_lineage_and_as_of_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            FixtureIngestor(store).ingest()

            CorpusBuilder(store).build(DataMixtureConfig("demo", [MixtureSource("filings", 1.0, "cpt_text")]))
            cpt = store.read_jsonl("stage=04_corpus/mixture=demo/cpt.jsonl")

            self.assertEqual(cpt[0]["source_ids"], ["doc-acme-2022-10k"])
            self.assertEqual(cpt[0]["as_of_time"], "2023-02-15T16:30:00Z")

    def test_future_evidence_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            FixtureIngestor(store).ingest()
            future_row = store.read_jsonl("stage=02_normalized/family=evidence/source=fixture.jsonl")[0]
            future_row["timestamp"] = "2025-01-01T00:00:00Z"
            store.write_jsonl("stage=02_normalized/family=evidence/source=fixture.jsonl", [future_row])

            with self.assertRaises(LeakageError):
                CorpusBuilder(store).build(DataMixtureConfig("demo", [MixtureSource("qa", 1.0, "sft_instruction")]))


if __name__ == "__main__":
    unittest.main()
