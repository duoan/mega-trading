import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from mega_trading.core.store import LocalObjectStore
from mega_trading.train.config import TradingFoundationTrainConfig
from mega_trading.train.dataset import TradingFoundationDataset
from mega_trading.train.model import (
    EarningsEncoder,
    SecFilingEncoder,
    ForwardReturnDecoder,
    GatedCrossAttentionFusion,
    MacroEncoder,
    NewsEncoder,
    MarketDataEncoder,
    RiskDecoder,
    SharedMarketMemory,
    SwiGLU,
    TradingFoundationModel,
)
from mega_trading.train.trainer import TradingFoundationTrainer, _resolve_device, _resolve_precision


class TradingFoundationModelTests(unittest.TestCase):
    def test_dataset_pads_streams_and_encodes_labels(self) -> None:
        rows = [_row("a", "outperform", "low"), _row("b", "underperform", "high", sec_filings=[1.0, 2.0])]

        dataset = TradingFoundationDataset(rows, market_window_size=4, news_size=2, sec_filing_size=3, earnings_size=2, macro_size=2)
        item = dataset[0]

        self.assertEqual(item["market_data"].shape, torch.Size([4, 2]))
        self.assertEqual(item["news"].shape, torch.Size([2]))
        self.assertEqual(item["sec_filings"].shape, torch.Size([3]))
        self.assertEqual(item["earnings"].shape, torch.Size([2]))
        self.assertEqual(item["macro"].shape, torch.Size([2]))
        self.assertEqual(item["return_label"].item(), 2)
        self.assertEqual(item["risk_label"].item(), 0)

    def test_model_forward_returns_two_prediction_heads(self) -> None:
        model = TradingFoundationModel(market_window_size=4, news_size=2, sec_filing_size=3, earnings_size=2, macro_size=2, hidden_dim=8)
        batch = {
            "market_data": torch.zeros((2, 4, 2), dtype=torch.float32),
            "news": torch.zeros((2, 2), dtype=torch.float32),
            "sec_filings": torch.zeros((2, 3), dtype=torch.float32),
            "earnings": torch.zeros((2, 2), dtype=torch.float32),
            "macro": torch.zeros((2, 2), dtype=torch.float32),
        }

        return_logits, risk_logits = model(batch)

        self.assertEqual(return_logits.shape, torch.Size([2, 3]))
        self.assertEqual(risk_logits.shape, torch.Size([2, 3]))
        self.assertIsInstance(model.market_data_encoder, MarketDataEncoder)
        self.assertIsInstance(model.news_encoder, NewsEncoder)
        self.assertIsInstance(model.sec_filing_encoder, SecFilingEncoder)
        self.assertIsInstance(model.earnings_encoder, EarningsEncoder)
        self.assertIsInstance(model.macro_encoder, MacroEncoder)
        self.assertIsInstance(model.fusion, GatedCrossAttentionFusion)
        self.assertIsInstance(model.market_memory, SharedMarketMemory)
        self.assertIsInstance(model.forward_return_decoder, ForwardReturnDecoder)
        self.assertIsInstance(model.risk_decoder, RiskDecoder)
        self.assertEqual(model.forward_return_decoder.return_bucket_head.out_features, 3)
        self.assertEqual(model.risk_decoder.risk_bucket_head.out_features, 3)
        self.assertTrue(model.use_news)
        self.assertTrue(model.use_sec_filings)
        self.assertTrue(model.use_earnings)
        self.assertTrue(model.use_macro)
        self.assertEqual(model.fusion.cross_attention.num_heads, 4)
        self.assertEqual(model.market_memory.memory.shape, torch.Size([4, 8]))
        self.assertIsInstance(model.fusion.output_ffn, SwiGLU)
        self.assertEqual(model.fusion.output_ffn.w1.in_features, 8)
        self.assertEqual(model.fusion.output_ffn.w1.out_features, 32)
        self.assertEqual(model.fusion.output_ffn.w2.in_features, 32)
        self.assertEqual(model.fusion.output_ffn.w2.out_features, 8)
        self.assertEqual(model.fusion.output_ffn.w3.in_features, 8)
        self.assertEqual(model.fusion.output_ffn.w3.out_features, 32)

    def test_model_supports_modality_ablation(self) -> None:
        model = TradingFoundationModel(
            market_window_size=4,
            news_size=2,
            sec_filing_size=3,
            earnings_size=2,
            macro_size=2,
            hidden_dim=8,
            use_market_data=True,
            use_news=False,
            use_sec_filings=False,
            use_earnings=False,
            use_macro=False,
        )
        batch = {
            "market_data": torch.zeros((2, 4, 2), dtype=torch.float32),
            "news": torch.ones((2, 2), dtype=torch.float32),
            "sec_filings": torch.ones((2, 3), dtype=torch.float32),
            "earnings": torch.ones((2, 2), dtype=torch.float32),
            "macro": torch.ones((2, 2), dtype=torch.float32),
        }

        return_logits, risk_logits = model(batch)

        self.assertEqual(return_logits.shape, torch.Size([2, 3]))
        self.assertEqual(risk_logits.shape, torch.Size([2, 3]))

    def test_model_requires_hidden_dim_divisible_by_attention_heads(self) -> None:
        with self.assertRaises(ValueError):
            TradingFoundationModel(
                market_window_size=4,
                news_size=2,
                sec_filing_size=3,
                earnings_size=2,
                macro_size=2,
                hidden_dim=10,
                attention_heads=4,
            )

    def test_config_requires_at_least_one_modality(self) -> None:
        with self.assertRaises(ValueError):
            TradingFoundationTrainConfig(
                run_id="bad",
                max_steps=1,
                use_market_data=False,
                use_news=False,
                use_sec_filings=False,
                use_earnings=False,
                use_macro=False,
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
        with self.assertRaises(ValueError):
            TradingFoundationTrainConfig(run_id="bad", max_steps=1, wandb_mode="local")

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
            self.assertEqual(manifest.metadata["stream_contract"], "market_data_news_sec_filings_earnings_macro")
            self.assertEqual(manifest.metadata["use_market_data"], "True")
            self.assertEqual(manifest.metadata["use_news"], "True")
            self.assertEqual(manifest.metadata["use_sec_filings"], "True")
            self.assertEqual(manifest.metadata["use_earnings"], "True")
            self.assertEqual(manifest.metadata["use_macro"], "True")
            self.assertIn("model_version_id", manifest.metadata)
            self.assertIn("model_version_path", manifest.metadata)
            self.assertTrue((Path(tmp) / manifest.metadata["model_version_path"]).exists())
            self.assertEqual(manifest.metadata["requested_device"], "cpu")
            self.assertEqual(manifest.metadata["device"], "cpu")
            self.assertEqual(manifest.metadata["requested_precision"], "auto")
            self.assertEqual(manifest.metadata["precision"], "fp32")
            self.assertEqual(manifest.metadata["training_backend"], "accelerate")
            self.assertEqual(manifest.metadata["accelerator_mixed_precision"], "no")
            self.assertEqual(manifest.metadata["accelerator_num_processes"], "1")
            self.assertEqual(manifest.metadata["wandb_enabled"], "True")
            self.assertEqual(manifest.metadata["wandb_project"], "mega-trading")
            self.assertEqual(manifest.metadata["wandb_mode"], "offline")
            self.assertTrue((Path(tmp) / "runs/tfm-test/wandb").exists())


def _row(
    sample_id: str,
    return_label: str,
    risk_label: str,
    sec_filings: list[float] | None = None,
    as_of_time: str = "2024-01-02T00:00:00Z",
) -> dict[str, object]:
    return {
        "sample_id": f"sample-{sample_id}",
        "ticker": "ACME",
        "as_of_time": as_of_time,
        "market_returns": [0.0, 0.01, -0.02],
        "market_levels": [0.0, 0.01, -0.01],
        "news_embeddings": [0.2, -0.1],
        "sec_filing_features": sec_filings or [100.0],
        "earnings_features": [],
        "macro_features": [0.5],
        "return_label": return_label,
        "risk_label": risk_label,
        "source_ids": [sample_id],
    }


if __name__ == "__main__":
    unittest.main()
