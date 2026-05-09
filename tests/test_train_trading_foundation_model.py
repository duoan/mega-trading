import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from mega_trading.core.store import LocalObjectStore
from mega_trading.train.config import TradingFoundationTrainConfig
from mega_trading.train.dataset import TradingFoundationDataset
from mega_trading.train.model import SwiGLU, TradingFoundationModel
from mega_trading.train.trainer import TradingFoundationTrainer, _resolve_device, _resolve_precision


class TradingFoundationModelTests(unittest.TestCase):
    def test_dataset_pads_streams_and_encodes_labels(self) -> None:
        rows = [_row("a", "outperform", "low"), _row("b", "underperform", "high", fundamentals=[1.0, 2.0])]

        dataset = TradingFoundationDataset(rows, price_window_size=4, fundamental_size=3, evidence_size=5)
        item = dataset[0]

        self.assertEqual(item["price"].shape, torch.Size([4, 2]))
        self.assertEqual(item["fundamentals"].shape, torch.Size([3]))
        self.assertEqual(item["evidence"].shape, torch.Size([5]))
        self.assertEqual(item["return_label"].item(), 2)
        self.assertEqual(item["risk_label"].item(), 0)

    def test_model_forward_returns_two_prediction_heads(self) -> None:
        model = TradingFoundationModel(price_window_size=4, fundamental_size=3, evidence_size=5, hidden_dim=8)
        batch = {
            "price": torch.zeros((2, 4, 2), dtype=torch.float32),
            "fundamentals": torch.zeros((2, 3), dtype=torch.float32),
            "evidence": torch.zeros((2, 5), dtype=torch.float32),
        }

        return_logits, risk_logits = model(batch)

        self.assertEqual(return_logits.shape, torch.Size([2, 3]))
        self.assertEqual(risk_logits.shape, torch.Size([2, 3]))
        self.assertEqual(model.cross_attention.num_heads, 4)
        self.assertIsInstance(model.output_ffn, SwiGLU)
        self.assertEqual(model.output_ffn.w1.in_features, 8)
        self.assertEqual(model.output_ffn.w1.out_features, 32)
        self.assertEqual(model.output_ffn.w2.in_features, 32)
        self.assertEqual(model.output_ffn.w2.out_features, 8)
        self.assertEqual(model.output_ffn.w3.in_features, 8)
        self.assertEqual(model.output_ffn.w3.out_features, 32)

    def test_model_supports_modality_ablation(self) -> None:
        model = TradingFoundationModel(
            price_window_size=4,
            fundamental_size=3,
            evidence_size=5,
            hidden_dim=8,
            use_price=True,
            use_fundamentals=False,
            use_evidence=False,
        )
        batch = {
            "price": torch.zeros((2, 4, 2), dtype=torch.float32),
            "fundamentals": torch.ones((2, 3), dtype=torch.float32),
            "evidence": torch.ones((2, 5), dtype=torch.float32),
        }

        return_logits, risk_logits = model(batch)

        self.assertEqual(return_logits.shape, torch.Size([2, 3]))
        self.assertEqual(risk_logits.shape, torch.Size([2, 3]))

    def test_model_requires_hidden_dim_divisible_by_attention_heads(self) -> None:
        with self.assertRaises(ValueError):
            TradingFoundationModel(
                price_window_size=4,
                fundamental_size=3,
                evidence_size=5,
                hidden_dim=10,
                attention_heads=4,
            )

    def test_config_requires_at_least_one_modality(self) -> None:
        with self.assertRaises(ValueError):
            TradingFoundationTrainConfig(
                run_id="bad",
                max_steps=1,
                use_price=False,
                use_fundamentals=False,
                use_evidence=False,
            )

    def test_config_requires_attention_heads_to_divide_hidden_dim(self) -> None:
        with self.assertRaises(ValueError):
            TradingFoundationTrainConfig(run_id="bad", max_steps=1, hidden_dim=10, attention_heads=4)

    def test_config_requires_valid_validation_settings(self) -> None:
        with self.assertRaises(ValueError):
            TradingFoundationTrainConfig(run_id="bad", max_steps=1, validation_fraction=1.0)
        with self.assertRaises(ValueError):
            TradingFoundationTrainConfig(run_id="bad", max_steps=1, eval_interval=0)

    def test_config_requires_valid_device_and_precision(self) -> None:
        with self.assertRaises(ValueError):
            TradingFoundationTrainConfig(run_id="bad", max_steps=1, device="tpu")
        with self.assertRaises(ValueError):
            TradingFoundationTrainConfig(run_id="bad", max_steps=1, precision="bf16")

    def test_auto_device_prefers_cuda_then_mps_then_cpu(self) -> None:
        with patch("torch.cuda.is_available", return_value=True), patch(
            "mega_trading.train.trainer._mps_available", return_value=True
        ):
            self.assertEqual(_resolve_device("auto").type, "cuda")
        with patch("torch.cuda.is_available", return_value=False), patch(
            "mega_trading.train.trainer._mps_available", return_value=True
        ):
            self.assertEqual(_resolve_device("auto").type, "mps")
        with patch("torch.cuda.is_available", return_value=False), patch(
            "mega_trading.train.trainer._mps_available", return_value=False
        ):
            self.assertEqual(_resolve_device("auto").type, "cpu")

    def test_auto_precision_uses_mixed_only_on_cuda(self) -> None:
        self.assertEqual(_resolve_precision("auto", torch.device("cuda")), "mixed")
        self.assertEqual(_resolve_precision("auto", torch.device("mps")), "fp32")
        self.assertEqual(_resolve_precision("auto", torch.device("cpu")), "fp32")
        with self.assertRaises(RuntimeError):
            _resolve_precision("mixed", torch.device("mps"))

    def test_trainer_consumes_stream_shards_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            store.write_jsonl(
                "stage=05_shards/mixture=public/samples.jsonl",
                [
                    _row("a", "outperform", "low", as_of_time="2024-01-02T00:00:00Z"),
                    _row("b", "underperform", "high", as_of_time="2024-01-03T00:00:00Z"),
                    _row("c", "neutral", "medium", as_of_time="2024-01-04T00:00:00Z"),
                    _row("d", "outperform", "high", as_of_time="2024-01-05T00:00:00Z"),
                ],
            )

            result = TradingFoundationTrainer(
                store,
                TradingFoundationTrainConfig(
                    run_id="tfm-test",
                    max_steps=2,
                    hidden_dim=8,
                    batch_size=2,
                    validation_fraction=0.25,
                    eval_interval=1,
                    device="cpu",
                ),
            ).train("stage=05_shards/mixture=public/samples.jsonl")
            metrics = store.read_jsonl("runs/tfm-test/metrics.jsonl")
            manifest = store.read_manifest(result.manifest_path)

            self.assertEqual(result.steps, 2)
            self.assertEqual(metrics[-1]["stage"], "trading_foundation_model")
            self.assertIn("loss", metrics[-1])
            self.assertIn("validation_loss", metrics[-1])
            self.assertEqual(metrics[-1]["train_sample_count"], 3)
            self.assertEqual(metrics[-1]["validation_sample_count"], 1)
            self.assertTrue((Path(tmp) / result.checkpoint_path).exists())
            self.assertEqual(manifest.metadata["stage"], "trading_foundation_model")
            self.assertEqual(manifest.metadata["attention_heads"], "4")
            self.assertEqual(manifest.metadata["train_sample_count"], "3")
            self.assertEqual(manifest.metadata["validation_sample_count"], "1")
            self.assertEqual(manifest.metadata["use_price"], "True")
            self.assertIn("model_version_id", manifest.metadata)
            self.assertIn("model_version_path", manifest.metadata)
            self.assertTrue((Path(tmp) / manifest.metadata["model_version_path"]).exists())
            self.assertEqual(manifest.metadata["requested_device"], "cpu")
            self.assertEqual(manifest.metadata["device"], "cpu")
            self.assertEqual(manifest.metadata["requested_precision"], "auto")
            self.assertEqual(manifest.metadata["precision"], "fp32")


def _row(
    sample_id: str,
    return_label: str,
    risk_label: str,
    fundamentals: list[float] | None = None,
    as_of_time: str = "2024-01-02T00:00:00Z",
) -> dict[str, object]:
    return {
        "sample_id": f"sample-{sample_id}",
        "ticker": "ACME",
        "as_of_time": as_of_time,
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
