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
from mega_trading.model import TradingModel
from mega_trading.tokenizer import MarketEventTokenizer
from mega_trading.trainer import Trainer


class TrainingTests(unittest.TestCase):
    def test_event_builder_writes_token_shards_and_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_market_data(root)
            store = LocalObjectStore(root)

            result = EventBuilder(
                store,
                BuildConfig(block_size=8, stride=4, min_events_per_ticker=5),
            ).build()
            profile = json.loads(root.joinpath(result.profile_path).read_text(encoding="utf-8"))
            first_sequence = json.loads(root.joinpath(result.shard_path).read_text(encoding="utf-8").splitlines()[0])

            self.assertEqual(profile["stream_contract"], "market-event-token-v1")
            self.assertGreater(profile["event_count"], 10)
            self.assertGreater(profile["sequence_count"], 1)
            self.assertEqual(len(first_sequence["tokens"]), 9)

    def test_tokenizer_is_deterministic_and_model_forward_shapes(self) -> None:
        tokenizer = MarketEventTokenizer()
        event = {
            "side": "buy",
            "return_bps": 42.0,
            "range_bps": 80.0,
            "gap_bps": 5.0,
            "log_volume_ratio": 0.25,
            "dt_days": 1,
            "weekday": 2,
        }
        tokens = tokenizer.encode_event(event)
        model = TradingModel(vocab_size=tokenizer.vocab_size, block_size=8, hidden_dim=16, layers=1, attention_heads=2)
        logits = model(torch.tensor([tokens[:8]], dtype=torch.long))

        self.assertEqual(tokens, tokenizer.encode_event(event))
        self.assertEqual(logits.shape, (1, len(tokens[:8]), tokenizer.vocab_size))

    def test_dataset_row_maps_to_next_token_example(self) -> None:
        example = row_to_example({"tokens": [1, 3, 4, 5]})

        self.assertEqual(example["input_ids"].tolist(), [1, 3, 4])
        self.assertEqual(example["labels"].tolist(), [3, 4, 5])

    def test_trainer_and_eval_smoke_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_market_data(root)
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
            _write_market_data(root)

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


def _write_market_data(root: Path) -> None:
    target = root / "stage=02_normalized/family=market_data/source=yahoo.jsonl"
    target.parent.mkdir(parents=True)
    rows = []
    for ticker_index, ticker in enumerate(("AAPL", "MSFT")):
        base = 100.0 + ticker_index * 20.0
        for day in range(1, 16):
            close = base + day * (1.0 + 0.1 * ticker_index) + ((-1) ** day) * 0.3
            rows.append(
                {
                    "market_data_id": f"yahoo-{ticker}-2024-01-{day:02d}",
                    "ticker": ticker,
                    "date": f"2024-01-{day:02d}",
                    "open": close - 0.2,
                    "high": close + 0.8,
                    "low": close - 0.7,
                    "adjusted_close": close,
                    "volume": 1_000_000 + day * 10_000 + ticker_index * 5_000,
                    "provider": "yahoo",
                    "source_ids": [f"yahoo-{ticker}-2024-01-{day:02d}"],
                }
            )
    target.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
