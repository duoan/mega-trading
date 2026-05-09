import json
import tempfile
import unittest
from pathlib import Path

import torch

from mega_trading.cli import main
from mega_trading.config import BuildConfig, TrainConfig
from mega_trading.core.store import LocalObjectStore
from mega_trading.dataset import row_to_example
from mega_trading.eval import run_eval
from mega_trading.events import EventBuilder
from mega_trading.model import LlamaAttention, RMSNorm, SwiGLU, TradingModel
from mega_trading.tokenizer import MarketEventTokenizer
from mega_trading.trainer import Trainer


class TrainingTests(unittest.TestCase):
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

            self.assertEqual(profile["stream_contract"], "paper-order-flow-token-v1")
            self.assertEqual(profile["feature_order"], ["action", "side", "relative_price", "price_depth", "size", "time"])
            self.assertEqual(profile["event_size"], 1)
            self.assertEqual(tokenizer["type"], "paper-order-flow-composite")
            self.assertGreater(profile["event_count"], 10)
            self.assertGreater(profile["sequence_count"], 1)
            self.assertEqual(len(first_sequence["tokens"]), 9)

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

    def test_dataset_row_maps_to_next_token_example(self) -> None:
        example = row_to_example({"tokens": [1, 3, 4, 5]})

        self.assertEqual(example["input_ids"].tolist(), [1, 3, 4])
        self.assertEqual(example["labels"].tolist(), [3, 4, 5])

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
                ),
            ).train("stage=05_shards/mixture=public/tokens.jsonl")
            eval_result = run_eval(store, "train-test", rollouts=2, generated_tokens=8, device="cpu")

            metrics = root.joinpath(train_result.metrics_path).read_text(encoding="utf-8")
            report = json.loads(root.joinpath(eval_result.report_path).read_text(encoding="utf-8"))
            self.assertTrue(root.joinpath(train_result.checkpoint_path).exists())
            self.assertIn("validation_loss", metrics)
            self.assertEqual(report["stage"], "eval")
            self.assertIn("generated", report)

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
