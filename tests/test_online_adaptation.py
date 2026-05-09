import tempfile
import unittest
from pathlib import Path

from mega_trading.core.store import LocalObjectStore
from mega_trading.eval.replay_buffer import ReplayBufferConfig, select_replay_batch
from mega_trading.train.online import OnlineAdaptationConfig, run_online_adapter_update


class OnlineAdaptationTests(unittest.TestCase):
    def test_replay_buffer_mixes_recent_same_regime_and_random_rows(self) -> None:
        rows = [_row(index, "low" if index % 2 == 0 else "high") for index in range(10)]

        batch = select_replay_batch(rows, ReplayBufferConfig(max_samples=10, seed=1), target_regime="low")

        self.assertEqual(len(batch), 10)
        self.assertEqual([row["prediction_id"] for row in batch[:7]], [f"pred-{index}" for index in range(3, 10)])
        self.assertTrue(any(row["actual_risk_bucket"] == "low" for row in batch[7:9]))

    def test_online_adapter_update_writes_stub_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            labeled_path = "labels/demo/labeled-predictions.jsonl"
            store.write_jsonl(labeled_path, [_row(index, "low") for index in range(4)])

            result = run_online_adapter_update(
                store,
                update_id="update-1",
                labeled_prediction_path=labeled_path,
                base_model_version="base-v1",
                config=OnlineAdaptationConfig(replay_buffer=ReplayBufferConfig(max_samples=3)),
            )
            adapter = store.read_json(result.adapter_path)
            manifest = store.read_manifest(result.manifest_path)

            self.assertEqual(result.samples, 3)
            self.assertEqual(adapter["status"], "stub")
            self.assertIn("adapter", adapter["trainable_components"])
            self.assertEqual(manifest.metadata["samples"], "3")


def _row(index: int, risk: str) -> dict[str, object]:
    return {
        "prediction_id": f"pred-{index}",
        "prediction_time": f"2024-01-{index + 1:02d}T00:00:00Z",
        "ticker": "ACME",
        "sample_id": f"sample-{index}",
        "label_status": "ready",
        "actual_risk_bucket": risk,
        "actual_return_bucket": "outperform",
        "actual_forward_return": 0.01,
    }


if __name__ == "__main__":
    unittest.main()
