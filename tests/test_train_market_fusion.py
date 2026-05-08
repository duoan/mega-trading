import tempfile
import unittest
from pathlib import Path

import torch

from marketfm.core.store import LocalObjectStore
from marketfm.train.market_fusion import (
    MarketFusionDataset,
    MarketFusionModel,
    MarketFusionTrainConfig,
    MarketFusionTrainer,
)


class MarketFusionTests(unittest.TestCase):
    def test_dataset_pads_streams_and_encodes_labels(self) -> None:
        rows = [_row("a", "outperform", "low"), _row("b", "underperform", "high", fundamentals=[1.0, 2.0])]

        dataset = MarketFusionDataset(rows, price_window_size=4, fundamental_size=3, evidence_size=5)
        item = dataset[0]

        self.assertEqual(item["price"].shape, torch.Size([4, 2]))
        self.assertEqual(item["fundamentals"].shape, torch.Size([3]))
        self.assertEqual(item["evidence"].shape, torch.Size([5]))
        self.assertEqual(item["return_label"].item(), 2)
        self.assertEqual(item["risk_label"].item(), 0)

    def test_model_forward_returns_two_prediction_heads(self) -> None:
        model = MarketFusionModel(price_window_size=4, fundamental_size=3, evidence_size=5, hidden_dim=8)
        batch = {
            "price": torch.zeros((2, 4, 2), dtype=torch.float32),
            "fundamentals": torch.zeros((2, 3), dtype=torch.float32),
            "evidence": torch.zeros((2, 5), dtype=torch.float32),
        }

        return_logits, risk_logits = model(batch)

        self.assertEqual(return_logits.shape, torch.Size([2, 3]))
        self.assertEqual(risk_logits.shape, torch.Size([2, 3]))

    def test_trainer_consumes_stream_shards_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            store.write_jsonl(
                "stage=05_shards/mixture=public/samples.jsonl",
                [_row("a", "outperform", "low"), _row("b", "underperform", "high")],
            )

            result = MarketFusionTrainer(
                store,
                MarketFusionTrainConfig(run_id="market-fusion-test", max_steps=2, hidden_dim=8, batch_size=2),
            ).train("stage=05_shards/mixture=public/samples.jsonl")
            metrics = store.read_jsonl("runs/market-fusion-test/metrics.jsonl")
            manifest = store.read_manifest(result.manifest_path)

            self.assertEqual(result.steps, 2)
            self.assertEqual(metrics[-1]["stage"], "market_fusion")
            self.assertIn("loss", metrics[-1])
            self.assertTrue((Path(tmp) / result.checkpoint_path).exists())
            self.assertEqual(manifest.metadata["stage"], "market_fusion")


def _row(
    sample_id: str,
    return_label: str,
    risk_label: str,
    fundamentals: list[float] | None = None,
) -> dict[str, object]:
    return {
        "sample_id": f"sample-{sample_id}",
        "ticker": "ACME",
        "as_of_time": "2024-01-02T00:00:00Z",
        "price_returns": [0.0, 0.01, -0.02],
        "price_levels": [0.0, 0.01, -0.01],
        "fundamental_values": fundamentals or [100.0],
        "evidence_token_ids": [2, 3, 4],
        "return_label": return_label,
        "risk_label": risk_label,
        "source_ids": [sample_id],
        "evidence_ids": [],
    }


if __name__ == "__main__":
    unittest.main()
