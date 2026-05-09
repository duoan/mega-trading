import tempfile
import unittest
from pathlib import Path

from mega_trading.core.store import LocalObjectStore
from mega_trading.eval.delayed_labels import materialize_delayed_labels
from mega_trading.eval.replay import run_replay_inference
from mega_trading.train.config import TradingFoundationTrainConfig
from mega_trading.train.trainer import TradingFoundationTrainer


class ReplayInferenceTests(unittest.TestCase):
    def test_replay_inference_writes_prediction_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            shard_path = "stage=05_shards/mixture=public/samples.jsonl"
            store.write_jsonl(
                shard_path,
                [
                    _row("a", "outperform", "low"),
                    _row("b", "underperform", "high"),
                    _row("c", "neutral", "medium"),
                ],
            )
            train_result = TradingFoundationTrainer(
                store,
                TradingFoundationTrainConfig(
                    run_id="tfm-replay",
                    max_steps=1,
                    hidden_dim=8,
                    batch_size=2,
                    validation_fraction=0.0,
                    device="cpu",
                ),
            ).train(shard_path)
            train_manifest = store.read_manifest(train_result.manifest_path)

            result = run_replay_inference(
                store,
                replay_id="replay-smoke",
                model_version_id=str(train_manifest.metadata["model_version_id"]),
                shard_path=shard_path,
                max_predictions=2,
                device="cpu",
            )
            predictions = store.read_jsonl(result.prediction_path)
            manifest = store.read_manifest(result.manifest_path)

            self.assertEqual(result.predictions, 2)
            self.assertEqual(len(predictions), 2)
            self.assertEqual(predictions[0]["label_status"], "pending")
            self.assertEqual(predictions[0]["model_version_id"], train_manifest.metadata["model_version_id"])
            self.assertIn("confidence", predictions[0])
            self.assertEqual(manifest.metadata["predictions"], "2")

            label_result = materialize_delayed_labels(
                store,
                label_run_id="labels-smoke",
                prediction_path=result.prediction_path,
                shard_path=shard_path,
            )
            labels = store.read_jsonl(label_result.label_path)
            labeled_predictions = store.read_jsonl(label_result.labeled_prediction_path)

            self.assertEqual(label_result.labels, 2)
            self.assertEqual(labels[0]["actual_return_bucket"], "outperform")
            self.assertEqual(labeled_predictions[0]["label_status"], "ready")
            self.assertEqual(labeled_predictions[0]["actual_risk_bucket"], "low")


def _row(sample_id: str, return_label: str, risk_label: str) -> dict[str, object]:
    return {
        "sample_id": f"sample-{sample_id}",
        "ticker": "ACME",
        "as_of_time": "2024-01-02T00:00:00Z",
        "price_returns": [0.0, 0.01, -0.02],
        "price_levels": [0.0, 0.01, -0.01],
        "fundamental_values": [100.0],
        "evidence_token_ids": [2, 3, 4],
        "return_label": return_label,
        "risk_label": risk_label,
        "forward_return": 0.01,
        "label_ready_time": "2024-01-22T00:00:00Z",
        "source_ids": [sample_id],
        "evidence_ids": [],
    }


if __name__ == "__main__":
    unittest.main()
