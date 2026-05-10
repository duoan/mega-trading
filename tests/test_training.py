import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from mega_trading.cli import load_config, main
from mega_trading.config import BuildConfig, TrainConfig
from mega_trading.core.store import LocalObjectStore
from mega_trading.dataset import NumpyTickerTimeDataset, row_to_example
from mega_trading.eval import run_eval
from mega_trading.events import EventBuilder
from mega_trading.model import LlamaAttention, RMSNorm, SwiGLU, TradingModel
from mega_trading.tokenizer import MarketEventTokenizer
from mega_trading.trainer import Trainer, _attention_kernel_context, _maybe_compile_model


class TrainingTests(unittest.TestCase):
    def test_named_training_configs_parse_for_local_and_modal_runs(self) -> None:
        local = load_config(Path("configs"), "local-mac", [])
        binance_local = load_config(Path("configs"), "binance-local", [])
        binance_modal_prep = load_config(Path("configs"), "binance-modal-prep", [])
        binance_modal = load_config(Path("configs"), "modal-binance", [])
        proxy = load_config(Path("configs"), "modal-proxy", [])
        paper = load_config(Path("configs"), "modal-paper", [])

        self.assertEqual(str(local.data.data_dir), ".mega-trading/local-mac")
        self.assertEqual(int(local.model.hidden_dim), 192)
        self.assertEqual(str(local.training.device), "mps")
        self.assertEqual(str(binance_local.data.source), "binance_trades")
        self.assertEqual(str(binance_modal_prep.data.data_dir), ".mega-trading/binance-modal")
        self.assertEqual(str(binance_modal_prep.data.mixture), "binance_public")
        self.assertEqual(str(binance_modal.data.data_dir), "/data/binance-trades")
        self.assertEqual(str(binance_modal.data.mixture), "binance_public")
        self.assertEqual(str(binance_modal.training.distributed_strategy), "fsdp")
        self.assertEqual(str(proxy.data.data_dir), "/data/hf-1m-proxy")
        self.assertEqual(int(proxy.build.block_size), 512)
        self.assertEqual(str(proxy.training.distributed_strategy), "fsdp")
        self.assertEqual(str(paper.data.data_dir), "/data/hf-1m-paper")
        self.assertEqual(int(paper.build.block_size), 1024)
        self.assertEqual(int(paper.model.hidden_dim), 1536)
        self.assertEqual(int(paper.model.layers), 20)
        self.assertTrue(bool(paper.training.compile))
        self.assertEqual(str(paper.training.attention_backend), "flash")

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

    def test_event_builder_writes_token_shards_and_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_order_flow(root)
            store = LocalObjectStore(root)

            result = EventBuilder(
                store,
                BuildConfig(block_size=8, stride=4, min_events_per_ticker=5),
            ).build()
            profile = json.loads(root.joinpath(result.profile_path).read_text(encoding="utf-8"))
            tokenizer = json.loads(root.joinpath(result.tokenizer_path).read_text(encoding="utf-8"))
            first_sequence = json.loads(root.joinpath(result.shard_path).read_text(encoding="utf-8").splitlines()[0])
            tokens = np.load(root / result.numpy_tokens_path, mmap_mode="r")
            ticker_ids = np.load(root / result.numpy_ticker_ids_path, mmap_mode="r")
            numpy_metadata = json.loads(root.joinpath(result.numpy_metadata_path).read_text(encoding="utf-8"))

            self.assertEqual(profile["stream_contract"], "paper-order-flow-token-v1")
            self.assertEqual(profile["feature_order"], ["action", "side", "relative_price", "price_depth", "size", "time"])
            self.assertEqual(profile["event_size"], 1)
            self.assertEqual(tokenizer["type"], "paper-order-flow-composite")
            self.assertGreater(profile["event_count"], 10)
            self.assertGreater(profile["sequence_count"], 1)
            self.assertEqual(len(first_sequence["tokens"]), 9)
            self.assertEqual(tokens.shape, (profile["sequence_count"], profile["block_size"] + 1))
            self.assertEqual(ticker_ids.shape, (profile["sequence_count"],))
            self.assertEqual(tokens[0].tolist(), first_sequence["tokens"])
            self.assertEqual(profile["numpy_dataset"]["format"], "mega-trading-numpy-token-v1")
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
        example = row_to_example({"tokens": [1, 3, 4, 5]})

        self.assertEqual(example["input_ids"].tolist(), [1, 3, 4])
        self.assertEqual(example["labels"].tolist(), [3, 4, 5])

    def test_numpy_dataset_maps_mmap_rows_to_next_token_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_order_flow(root)
            store = LocalObjectStore(root)
            result = EventBuilder(
                store,
                BuildConfig(block_size=8, stride=4, min_events_per_ticker=5),
            ).build()
            profile = store.read_json(result.profile_path)
            dataset = NumpyTickerTimeDataset(
                store,
                dict(profile["numpy_dataset"]),
                {"AAPL": 1, "MSFT": 1},
                split="train",
            )
            example = next(iter(dataset))

            self.assertEqual(example["input_ids"].shape[0], 8)
            self.assertEqual(example["labels"].shape[0], 8)

    def test_trainer_and_eval_smoke_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_order_flow(root)
            store = LocalObjectStore(root)
            EventBuilder(
                store,
                BuildConfig(block_size=8, stride=4, min_events_per_ticker=5),
            ).build()

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
                    wandb_enabled=False,
                    progress_bar=False,
                    checkpoint_interval=1,
                ),
            ).train("stage=05_shards/mixture=public/tokens.jsonl")
            eval_result = run_eval(store, "train-test", rollouts=2, generated_tokens=8, device="cpu")

            metrics = [json.loads(line) for line in root.joinpath(train_result.metrics_path).read_text(encoding="utf-8").splitlines()]
            manifest = store.read_manifest(train_result.manifest_path)
            checkpoint = torch.load(root / train_result.checkpoint_path, map_location="cpu", weights_only=False)
            report = json.loads(root.joinpath(eval_result.report_path).read_text(encoding="utf-8"))
            self.assertTrue(root.joinpath(train_result.checkpoint_path).exists())
            self.assertTrue(root.joinpath("runs/train-test/checkpoints/step-000001.pt").exists())
            self.assertIn("validation_loss", metrics[-1])
            self.assertEqual(metrics[-1]["distributed_strategy"], "ddp")
            self.assertEqual(metrics[-1]["dataset_format"], "numpy")
            self.assertEqual(metrics[-1]["world_size"], 1)
            self.assertFalse(metrics[-1]["compile_enabled"])
            self.assertEqual(manifest.metadata["distributed_strategy"], "ddp")
            self.assertEqual(manifest.metadata["dataset_format"], "numpy")
            self.assertEqual(manifest.metadata["gradient_accumulation_steps"], 1)
            self.assertEqual(manifest.metadata["attention_backend"], "auto")
            self.assertIn("optimizer_state_dict", checkpoint)
            self.assertEqual(checkpoint["step"], 2)
            self.assertEqual(report["stage"], "eval")
            self.assertIn("generated", report)

    def test_trainer_resumes_from_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_order_flow(root)
            store = LocalObjectStore(root)
            EventBuilder(
                store,
                BuildConfig(block_size=8, stride=4, min_events_per_ticker=5),
            ).build()

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
                    wandb_enabled=False,
                    progress_bar=False,
                ),
            ).train("stage=05_shards/mixture=public/tokens.jsonl")
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
                    wandb_enabled=False,
                    progress_bar=False,
                    resume_from_checkpoint=first.checkpoint_path,
                ),
            ).train("stage=05_shards/mixture=public/tokens.jsonl")

            metrics = [json.loads(line) for line in root.joinpath(resumed.metrics_path).read_text(encoding="utf-8").splitlines()]
            checkpoint = torch.load(root / resumed.checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual([row["step"] for row in metrics], [1, 2])
            self.assertEqual(checkpoint["step"], 2)

    def test_cli_build_train_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_order_flow(root)

            self.assertEqual(
                main(
                    [
                        "build",
                        f"data.data_dir={root}",
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
                        "training.wandb_enabled=false",
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

            self.assertTrue((root / "stage=05_shards/mixture=public/tokens.jsonl").exists())
            self.assertTrue((root / "runs/train-cli/checkpoint.pt").exists())
            self.assertTrue((root / "evals/train-cli/report.json").exists())


def _write_order_flow(root: Path) -> None:
    target = root / "stage=02_normalized/family=order_flow/source=hf_ohlcv_1m.jsonl"
    target.parent.mkdir(parents=True)
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
                    "provider": "hf_ohlcv_1m",
                    "source_ids": [f"hf:{ticker}:{index:04d}"],
                    "midprice_return_bps": float((1 if index % 2 else -1) * (index % 5)),
                }
            )
    target.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
