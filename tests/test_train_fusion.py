import tempfile
import unittest
from pathlib import Path

from marketfm.core.store import LocalObjectStore
from marketfm.train.fusion import FusionTrainConfig, TinyFusionTrainer


class TinyFusionTrainerTests(unittest.TestCase):
    def test_training_consumes_stream_shards_and_writes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = _prepared_store(tmp)

            result = TinyFusionTrainer(store, FusionTrainConfig(run_id="fusion-test", max_steps=3)).train(
                "stage=05_shards/mixture=public/samples.jsonl"
            )
            metrics = store.read_jsonl("runs/fusion-test/metrics.jsonl")
            checkpoint = store.read_json(result.checkpoint_path)

            self.assertEqual(result.steps, 3)
            self.assertEqual(len(metrics), 3)
            self.assertEqual(metrics[0]["stage"], "fusion")
            self.assertIn("return_accuracy", metrics[-1])
            self.assertEqual(checkpoint["stage"], "fusion")

    def test_training_writes_manifest_with_stream_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = _prepared_store(tmp)

            result = TinyFusionTrainer(store, FusionTrainConfig(run_id="manifest-test", max_steps=1)).train(
                "stage=05_shards/mixture=public/samples.jsonl"
            )
            manifest = store.read_manifest(result.manifest_path)

            self.assertEqual(manifest.artifact_type, "training_run")
            self.assertEqual(manifest.metadata["stage"], "fusion")
            self.assertEqual(manifest.metadata["stream_contract"], "price_fundamental_text")


def _prepared_store(tmp: str) -> LocalObjectStore:
    store = LocalObjectStore(Path(tmp))
    store.write_jsonl(
        "stage=05_shards/mixture=public/samples.jsonl",
        [
            _row("sample-a", [0.0, 0.02], [0.0, 0.02], [100.0], "outperform", "low"),
            _row("sample-b", [0.0, -0.03], [0.0, -0.03], [20.0], "underperform", "high"),
        ],
    )
    return store


def _row(
    sample_id: str,
    price_returns: list[float],
    price_levels: list[float],
    fundamental_values: list[float],
    return_label: str,
    risk_label: str,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "ticker": "ACME",
        "as_of_time": "2024-01-02T00:00:00Z",
        "price_returns": price_returns,
        "price_levels": price_levels,
        "fundamental_concepts": ["Revenue"],
        "fundamental_values": fundamental_values,
        "evidence_token_ids": [2, 3],
        "return_label": return_label,
        "risk_label": risk_label,
        "forward_return": 0.05,
        "source_ids": [sample_id],
        "evidence_ids": [],
    }


if __name__ == "__main__":
    unittest.main()
