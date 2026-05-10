import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from mega_trading.cli import load_config, main
from mega_trading.config import BuildConfig, TrainConfig
from mega_trading.backtest import run_backtest
from mega_trading.core.store import LocalObjectStore
from mega_trading.dataset import NumpyTickerTimeDataset, tokens_to_example
from mega_trading.eval import run_eval
from mega_trading.events import EventBuilder, events_by_ticker_from_rows
from mega_trading.model import LlamaAttention, RMSNorm, SwiGLU, TradingModel
from mega_trading.tokenizer import MarketEventTokenizer
from mega_trading.trainer import Trainer, _attention_kernel_context, _maybe_compile_model


class TrainingTests(unittest.TestCase):
    def test_named_training_configs_parse_for_local_and_modal_runs(self) -> None:
        binance_local = load_config(Path("configs"), "binance-local", [])
        binance_modal_prep = load_config(Path("configs"), "binance-modal-prep", [])
        server = load_config(Path("configs"), "server-rtx6000", [])
        binance_modal = load_config(Path("configs"), "modal-binance", [])

        self.assertEqual(str(binance_local.data.source), "binance_trades")
        self.assertEqual(int(binance_local.build.numpy_partition_rows), 8192)
        self.assertEqual(str(binance_modal_prep.data.data_dir), ".mega-trading/binance-modal")
        self.assertEqual(str(binance_modal_prep.data.mixture), "binance_public")
        self.assertEqual(int(binance_modal_prep.build.numpy_partition_rows), 16384)
        self.assertEqual(float(binance_modal_prep.build.validation_fraction), 0.02)
        self.assertEqual(float(binance_modal_prep.build.backtest_fraction), 0.10)
        self.assertTrue(bool(binance_modal_prep.build.streaming_prepare))
        self.assertEqual(str(server.data.data_dir), ".mega-trading/binance-modal")
        self.assertEqual(str(server.training.device), "cuda")
        self.assertEqual(str(server.training.distributed_strategy), "ddp")
        self.assertEqual(int(server.model.hidden_dim), 1024)
        self.assertEqual(int(server.model.layers), 20)
        self.assertEqual(int(server.training.batch_size), 8)
        self.assertEqual(int(server.training.gradient_accumulation_steps), 8)
        self.assertEqual(int(server.training.max_eval_batches), 64)
        self.assertEqual(str(binance_modal.data.data_dir), "/data/binance-trades")
        self.assertEqual(str(binance_modal.data.mixture), "binance_public")
        self.assertEqual(str(binance_modal.training.distributed_strategy), "fsdp")
        self.assertEqual(int(binance_modal.build.numpy_partition_rows), 16384)
        self.assertTrue(bool(server.build.streaming_prepare))

    def test_train_config_validates_distributed_runtime_options(self) -> None:
        with self.assertRaisesRegex(ValueError, "distributed_strategy"):
            TrainConfig(run_id="bad", distributed_strategy="deepspeed")
        with self.assertRaisesRegex(ValueError, "gradient_accumulation_steps"):
            TrainConfig(run_id="bad", gradient_accumulation_steps=0)
        with self.assertRaisesRegex(ValueError, "compile_mode"):
            TrainConfig(run_id="bad", compile_mode="fastest")
        with self.assertRaisesRegex(ValueError, "attention_backend"):
            TrainConfig(run_id="bad", attention_backend="xformers")
        with self.assertRaisesRegex(ValueError, "checkpoint_interval"):
            TrainConfig(run_id="bad", checkpoint_interval=0)
        with self.assertRaisesRegex(ValueError, "max_eval_batches"):
            TrainConfig(run_id="bad", max_eval_batches=0)

    def test_build_config_validates_prepared_split_fractions(self) -> None:
        with self.assertRaisesRegex(ValueError, "validation_fraction"):
            BuildConfig(validation_fraction=-0.1)
        with self.assertRaisesRegex(ValueError, "backtest_fraction"):
            BuildConfig(backtest_fraction=1.0)
        with self.assertRaisesRegex(ValueError, "validation_fraction \\+ backtest_fraction"):
            BuildConfig(validation_fraction=0.5, backtest_fraction=0.5)

    def test_event_builder_writes_token_shards_and_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalObjectStore(root)

            result = EventBuilder(
                store,
                BuildConfig(block_size=8, stride=4, min_events_per_ticker=5),
            ).build_events(_order_flow_events())
            profile = json.loads(root.joinpath(result.profile_path).read_text(encoding="utf-8"))
            tokenizer = json.loads(root.joinpath(result.tokenizer_path).read_text(encoding="utf-8"))
            tokens = np.load(root / result.numpy_tokens_path, mmap_mode="r")
            ticker_ids = np.load(root / result.numpy_ticker_ids_path, mmap_mode="r")
            numpy_metadata = json.loads(root.joinpath(result.numpy_metadata_path).read_text(encoding="utf-8"))

            self.assertEqual(profile["stream_contract"], "paper-order-flow-token-v1")
            self.assertEqual(profile["feature_order"], ["action", "side", "relative_price", "price_depth", "size", "time"])
            self.assertEqual(profile["event_size"], 1)
            self.assertEqual(tokenizer["type"], "paper-order-flow-composite")
            self.assertGreater(profile["event_count"], 10)
            self.assertGreater(profile["sequence_count"], 1)
            self.assertEqual(tokens.shape, (profile["sequence_count"], profile["block_size"] + 1))
            self.assertEqual(ticker_ids.shape, (profile["sequence_count"],))
            self.assertEqual(profile["numpy_dataset"]["format"], "mega-trading-numpy-token-v1")
            self.assertEqual(profile["numpy_dataset"]["splits"]["method"], "per_ticker_time")
            self.assertGreater(profile["numpy_dataset"]["splits"]["totals"]["backtest"], 0)
            self.assertIn("start_time", profile["numpy_dataset"]["splits"]["time_ranges"]["AAPL"]["backtest"])
            self.assertEqual(numpy_metadata["tokens_path"], result.numpy_tokens_path)
            self.assertIn("AAPL", numpy_metadata["ticker_to_id"])

    def test_tokenizer_is_deterministic_and_model_forward_shapes(self) -> None:
        events = [
            {
                "side": "buy" if index % 2 else "sell",
                "action": "add" if index % 3 else "delete",
                "relative_price_bps": float(index - 5),
                "price_depth_bps": float(index + 1),
                "size": float(100 + index),
                "interarrival_seconds": 1.0,
            }
            for index in range(10)
        ]
        tokenizer = MarketEventTokenizer.fit(events, relative_price_bins=4, price_bins=4, size_bins=4, time_bins=2)
        event = {
            "action": "add",
            "side": "buy",
            "relative_price_bps": 4.2,
            "price_depth_bps": 4.2,
            "size": 128.0,
            "interarrival_seconds": 1.0,
        }
        tokens = tokenizer.encode_event(event)
        model = TradingModel(vocab_size=tokenizer.vocab_size, block_size=8, hidden_dim=16, layers=1, attention_heads=2)
        logits = model(torch.tensor([tokens[:8]], dtype=torch.long))

        self.assertEqual(tokens, tokenizer.encode_event(event))
        self.assertEqual(len(tokens), 1)
        self.assertIsNotNone(tokenizer.price_depth_value(tokens[0]))
        self.assertEqual(logits.shape, (1, len(tokens[:8]), tokenizer.vocab_size))

    def test_model_uses_llama_style_decoder_blocks(self) -> None:
        tokenizer = MarketEventTokenizer()
        model = TradingModel(
            vocab_size=tokenizer.vocab_size,
            block_size=8,
            hidden_dim=16,
            layers=1,
            attention_heads=4,
            kv_heads=2,
            intermediate_dim=32,
        )
        block = model.blocks[0]

        self.assertIsInstance(model.norm, RMSNorm)
        self.assertFalse(hasattr(model, "position_embedding"))
        self.assertIsInstance(block.attention, LlamaAttention)
        self.assertEqual(block.attention.kv_heads, 2)
        self.assertIsInstance(block.feed_forward, SwiGLU)

    def test_compile_and_attention_backend_helpers_are_config_driven(self) -> None:
        model = TradingModel(vocab_size=MarketEventTokenizer().vocab_size, block_size=8, hidden_dim=16, layers=1, attention_heads=2)
        config = TrainConfig(run_id="compile-test", compile=True, compile_mode="reduce-overhead")

        with patch("mega_trading.trainer.torch.compile", return_value="compiled") as compile_model:
            self.assertEqual(_maybe_compile_model(model, config, torch.device("cpu")), "compiled")
            compile_model.assert_called_once_with(model, mode="reduce-overhead")
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            with _attention_kernel_context("flash", torch.device("cpu")):
                pass

    def test_dataset_row_maps_to_next_token_example(self) -> None:
        example = tokens_to_example([1, 3, 4, 5])

        self.assertEqual(example["input_ids"].tolist(), [1, 3, 4])
        self.assertEqual(example["labels"].tolist(), [3, 4, 5])

    def test_numpy_dataset_maps_mmap_rows_to_next_token_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalObjectStore(root)
            result = EventBuilder(
                store,
                BuildConfig(block_size=8, stride=4, min_events_per_ticker=5),
            ).build_events(_order_flow_events())
            profile = store.read_json(result.profile_path)
            dataset = NumpyTickerTimeDataset(
                store,
                dict(profile["numpy_dataset"]),
                dict(profile["numpy_dataset"]["splits"]["counts"]),
                split="train",
            )
            example = next(iter(dataset))

            self.assertEqual(example["input_ids"].shape[0], 8)
            self.assertEqual(example["labels"].shape[0], 8)

    def test_partitioned_numpy_dataset_maps_rows_to_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalObjectStore(root)
            result = EventBuilder(
                store,
                BuildConfig(block_size=8, stride=4, min_events_per_ticker=5, numpy_partition_rows=2),
            ).build_events(_order_flow_events())
            profile = store.read_json(result.profile_path)
            numpy_metadata = dict(profile["numpy_dataset"])
            dataset = NumpyTickerTimeDataset(
                store,
                numpy_metadata,
                dict(numpy_metadata["splits"]["counts"]),
                split="train",
            )
            example = next(iter(dataset))

            self.assertTrue(numpy_metadata["partitioned"])
            self.assertGreater(len(numpy_metadata["partitions"]), 1)
            self.assertEqual(example["input_ids"].shape[0], 8)

    def test_numpy_dataset_exposes_chronological_backtest_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalObjectStore(root)
            result = EventBuilder(
                store,
                BuildConfig(block_size=8, stride=4, min_events_per_ticker=5, validation_fraction=0.25, backtest_fraction=0.25),
            ).build_events(_order_flow_events())
            profile = store.read_json(result.profile_path)
            numpy_metadata = dict(profile["numpy_dataset"])
            split_counts = dict(numpy_metadata["splits"]["counts"])
            train_rows = list(NumpyTickerTimeDataset(store, numpy_metadata, split_counts, split="train"))
            validation_rows = list(NumpyTickerTimeDataset(store, numpy_metadata, split_counts, split="validation"))
            backtest_rows = list(NumpyTickerTimeDataset(store, numpy_metadata, split_counts, split="backtest"))

            self.assertGreater(len(train_rows), 0)
            self.assertGreater(len(validation_rows), 0)
            self.assertGreater(len(backtest_rows), 0)
            self.assertEqual(
                len(train_rows) + len(validation_rows) + len(backtest_rows),
                int(profile["sequence_count"]),
            )

    def test_trainer_and_eval_smoke_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalObjectStore(root)
            EventBuilder(
                store,
                BuildConfig(block_size=8, stride=4, min_events_per_ticker=5),
            ).build_events(_order_flow_events())

            train_result = Trainer(
                store,
                TrainConfig(
                    run_id="train-test",
                    max_steps=2,
                    batch_size=2,
                    validation_fraction=0.25,
                    eval_interval=1,
                    hidden_dim=16,
                    layers=1,
                    attention_heads=2,
                    device="cpu",
                    mlflow_enabled=False,
                    progress_bar=False,
                    checkpoint_interval=1,
                    max_eval_batches=1,
                ),
            ).train("datasets/mixture=public/tokens.npy")
            eval_result = run_eval(store, "train-test", rollouts=2, generated_tokens=8, device="cpu")
            backtest_result = run_backtest(store, "train-test", rollouts=2, generated_tokens=8, max_batches=1, device="cpu")

            metrics = json.loads(root.joinpath(train_result.metrics_path).read_text(encoding="utf-8"))["metrics"]
            manifest = store.read_manifest(train_result.manifest_path)
            checkpoint = torch.load(root / train_result.checkpoint_path, map_location="cpu", weights_only=False)
            report = json.loads(root.joinpath(eval_result.report_path).read_text(encoding="utf-8"))
            backtest_report = json.loads(root.joinpath(backtest_result.report_path).read_text(encoding="utf-8"))
            self.assertTrue(root.joinpath(train_result.checkpoint_path).exists())
            self.assertTrue(root.joinpath("runs/train-test/checkpoints/step-000001.pt").exists())
            self.assertIn("validation_loss", metrics[-1])
            self.assertEqual(metrics[-1]["validation_batches"], 1.0)
            self.assertGreater(metrics[-1]["backtest_sequence_count"], 0)
            self.assertEqual(metrics[-1]["distributed_strategy"], "ddp")
            self.assertEqual(metrics[-1]["dataset_format"], "numpy")
            self.assertEqual(metrics[-1]["world_size"], 1)
            self.assertFalse(metrics[-1]["compile_enabled"])
            self.assertEqual(manifest.metadata["distributed_strategy"], "ddp")
            self.assertEqual(manifest.metadata["dataset_format"], "numpy")
            self.assertEqual(manifest.metadata["gradient_accumulation_steps"], 1)
            self.assertEqual(manifest.metadata["attention_backend"], "auto")
            self.assertEqual(manifest.metadata["max_eval_batches"], 1)
            self.assertGreater(manifest.metadata["backtest_sequence_count"], 0)
            self.assertIn("optimizer_state_dict", checkpoint)
            self.assertEqual(checkpoint["step"], 2)
            self.assertEqual(report["stage"], "eval")
            self.assertIn("generated", report)
            self.assertEqual(backtest_report["stage"], "backtest")
            self.assertGreater(backtest_report["backtest_tokens"], 0)

    def test_trainer_resumes_from_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalObjectStore(root)
            EventBuilder(
                store,
                BuildConfig(block_size=8, stride=4, min_events_per_ticker=5),
            ).build_events(_order_flow_events())

            first = Trainer(
                store,
                TrainConfig(
                    run_id="resume-test",
                    max_steps=1,
                    batch_size=2,
                    validation_fraction=0.25,
                    eval_interval=1,
                    hidden_dim=16,
                    layers=1,
                    attention_heads=2,
                    device="cpu",
                    mlflow_enabled=False,
                    progress_bar=False,
                ),
            ).train("datasets/mixture=public/tokens.npy")
            resumed = Trainer(
                store,
                TrainConfig(
                    run_id="resume-test",
                    max_steps=2,
                    batch_size=2,
                    validation_fraction=0.25,
                    eval_interval=1,
                    hidden_dim=16,
                    layers=1,
                    attention_heads=2,
                    device="cpu",
                    mlflow_enabled=False,
                    progress_bar=False,
                    resume_from_checkpoint=first.checkpoint_path,
                ),
            ).train("datasets/mixture=public/tokens.npy")

            metrics = json.loads(root.joinpath(resumed.metrics_path).read_text(encoding="utf-8"))["metrics"]
            checkpoint = torch.load(root / resumed.checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual([row["step"] for row in metrics], [1, 2])
            self.assertEqual(checkpoint["step"], 2)

    def test_cli_build_train_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ingest_config = root / "ingest-demo.toml"
            ingest_config.write_text(
                f"""
[ingest]
output_dir = "{root}"

[[ingest.sources]]
name = "fixture"
""".strip()
                + "\n",
                encoding="utf-8",
            )

            self.assertEqual(
                main(
                    [
                        "prepare",
                        "--ingest-config",
                        str(ingest_config),
                        "build.block_size=8",
                        "build.stride=4",
                        "build.min_events_per_ticker=5",
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "train",
                        f"data.data_dir={root}",
                        "run.run_id=train-cli",
                        "training.max_steps=1",
                        "training.batch_size=2",
                        "training.validation_fraction=0.25",
                        "training.eval_interval=1",
                        "training.device=cpu",
                        "training.mlflow_enabled=false",
                        "training.progress_bar=false",
                        "training.gradient_accumulation_steps=1",
                        "training.attention_backend=auto",
                        "model.hidden_dim=16",
                        "model.layers=1",
                        "model.attention_heads=2",
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "eval",
                        "--run-id",
                        "train-cli",
                        f"data.data_dir={root}",
                        "eval.rollouts=1",
                        "eval.generated_tokens=4",
                        "eval.device=cpu",
                    ]
                ),
                0,
            )

            self.assertTrue((root / "datasets/mixture=public/tokens.npy").exists())
            self.assertTrue((root / "runs/train-cli/checkpoint.pt").exists())
            self.assertTrue((root / "evals/train-cli/report.json").exists())


def _order_flow_events() -> dict[str, list[dict[str, object]]]:
    rows = []
    for ticker_index, ticker in enumerate(("AAPL", "MSFT")):
        for index in range(20):
            rows.append(
                {
                    "event_id": f"event-{ticker}-{index:04d}",
                    "ticker": ticker,
                    "timestamp": f"2024-01-02T14:{30 + index:02d}:00Z",
                    "date": "2024-01-02",
                    "action": "add" if index % 3 else "delete",
                    "side": "buy" if (index + ticker_index) % 2 == 0 else "sell",
                    "midprice": 100.0 + ticker_index * 10.0 + index * 0.01,
                    "relative_price_bps": float((1 if (index + ticker_index) % 2 == 0 else -1) * (1 + index % 8)),
                    "price_depth_bps": float(1 + (index % 8) * 2 + ticker_index),
                    "size": float(1.0 + index * 0.1 + ticker_index * 0.05),
                    "interarrival_seconds": 60.0,
                    "provider": "fixture",
                    "source_ids": [f"fixture:{ticker}:{index:04d}"],
                    "midprice_return_bps": float((1 if index % 2 else -1) * (index % 5)),
                }
            )
    return events_by_ticker_from_rows(rows)


if __name__ == "__main__":
    unittest.main()
