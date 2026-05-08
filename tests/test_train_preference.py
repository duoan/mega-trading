import tempfile
import unittest
from pathlib import Path

from marketfm.core.config import DataMixtureConfig, MixtureSource
from marketfm.data.corpus import CorpusBuilder
from marketfm.data.ingest import FixtureIngestor
from marketfm.core.store import LocalObjectStore
from marketfm.train.preference import PreferencePairLoader, PreferenceTrainConfig, PreferenceTrainer


def _prepared_store(tmp: str) -> LocalObjectStore:
    store = LocalObjectStore(Path(tmp))
    FixtureIngestor(store).ingest()
    CorpusBuilder(store).build(DataMixtureConfig("demo", [MixtureSource("preference", 1.0, "preference_pair")]))
    return store


class PreferenceTrainerTests(unittest.TestCase):
    def test_preference_pairs_preserve_chosen_rejected_and_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = _prepared_store(tmp)
            pairs = PreferencePairLoader(store).load("corpus/demo/preference.jsonl")

            self.assertEqual(len(pairs), 2)
            self.assertIn("[ev-acme-margin]", pairs[0].chosen)
            self.assertEqual(pairs[0].evidence_ids, ["ev-acme-margin"])

    def test_preference_smoke_run_writes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = _prepared_store(tmp)

            result = PreferenceTrainer(store, PreferenceTrainConfig(run_id="pref-test")).train("corpus/demo/preference.jsonl")
            metrics = store.read_jsonl("runs/pref-test/metrics.jsonl")

            self.assertEqual(result.pairs, 2)
            self.assertEqual(metrics[0]["stage"], "preference")
            self.assertEqual(metrics[0]["preference_pairs"], 2)
            self.assertGreater(metrics[0]["chosen_evidence_rate"], metrics[0]["rejected_evidence_rate"])

    def test_preference_smoke_run_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = _prepared_store(tmp)

            result = PreferenceTrainer(store, PreferenceTrainConfig(run_id="pref-manifest")).train("corpus/demo/preference.jsonl")
            manifest = store.read_manifest(result.manifest_path)

            self.assertEqual(manifest.artifact_type, "training_run")
            self.assertEqual(manifest.metadata["stage"], "preference")
            self.assertEqual(manifest.metadata["pairs"], "2")


if __name__ == "__main__":
    unittest.main()
