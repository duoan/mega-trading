import tempfile
import unittest
from pathlib import Path

from marketfm.core.config import DataMixtureConfig, MixtureSource
from marketfm.data.corpus import CorpusBuilder
from marketfm.data.ingest import FixtureIngestor
from marketfm.core.store import LocalObjectStore
from marketfm.data.tokenize import ShardBuilder
from marketfm.train.cpt import CPTTrainer, TrainConfig


def _prepared_store(tmp: str) -> LocalObjectStore:
    store = LocalObjectStore(Path(tmp))
    FixtureIngestor(store).ingest()
    CorpusBuilder(store).build(DataMixtureConfig("demo", [MixtureSource("filings", 1.0, "cpt_text")]))
    ShardBuilder(store, sequence_length=8).build("demo", "cpt")
    return store


class CPTTrainerTests(unittest.TestCase):
    def test_training_consumes_shard_dataset_and_writes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = _prepared_store(tmp)
            trainer = CPTTrainer(store, TrainConfig(run_id="cpt-test", max_steps=3))

            result = trainer.train("shards/demo/cpt.jsonl")
            metrics = store.read_jsonl("runs/cpt-test/metrics.jsonl")

            self.assertEqual(result.steps, 3)
            self.assertEqual(len(metrics), 3)
            self.assertIn("loss", metrics[0])

    def test_checkpoint_save_and_resume_restore_step_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = _prepared_store(tmp)
            first = CPTTrainer(store, TrainConfig(run_id="resume-test", max_steps=2)).train("shards/demo/cpt.jsonl")

            resumed = CPTTrainer(
                store,
                TrainConfig(run_id="resume-test", max_steps=4, resume_from=first.checkpoint_path),
            ).train("shards/demo/cpt.jsonl")

            self.assertEqual(resumed.steps, 4)
            self.assertTrue(resumed.resume_used)
            checkpoint = store.read_json(resumed.checkpoint_path)
            self.assertEqual(checkpoint["step"], 4)

    def test_training_writes_run_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = _prepared_store(tmp)

            result = CPTTrainer(store, TrainConfig(run_id="manifest-test", max_steps=1)).train("shards/demo/cpt.jsonl")
            manifest = store.read_manifest(result.manifest_path)

            self.assertEqual(manifest.artifact_type, "training_run")
            self.assertEqual(manifest.metadata["stage"], "cpt")
            self.assertEqual(manifest.metadata["steps"], "1")


if __name__ == "__main__":
    unittest.main()
