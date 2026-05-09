import tempfile
import unittest
from pathlib import Path

from mega_trading.core.registry import ModelRegistry, model_version_id_for
from mega_trading.core.store import ArtifactPaths, LocalObjectStore


class ModelRegistryTests(unittest.TestCase):
    def test_registers_and_reads_training_model_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            registry = ModelRegistry(store)

            record = registry.register_training_run(
                run_id="tfm",
                checkpoint_path="runs/tfm/checkpoint.pt",
                manifest_path="manifests/runs/tfm-trading-foundation-model.json",
                metrics_path="runs/tfm/metrics.jsonl",
                config_hash="abcdef1234567890",
                feature_version="features-v1",
                label_version="labels-v1",
                data_snapshot_version="stage=05_shards/mixture=public/samples.jsonl",
                metadata={"steps": 2},
            )
            loaded = registry.read_model_version(record.model_version_id)

            self.assertEqual(record, loaded)
            self.assertTrue((Path(tmp) / ArtifactPaths().model(record.model_version_id)).exists())
            self.assertEqual(record.base_model_version, f"base-{record.model_version_id}")
            self.assertEqual(record.adapter_version, "adapter-none")

    def test_model_version_id_uses_config_hash_prefix(self) -> None:
        self.assertEqual(model_version_id_for("run-a", "abcdef1234567890"), "run-a-abcdef123456")


if __name__ == "__main__":
    unittest.main()
