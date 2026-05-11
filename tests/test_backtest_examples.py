import tempfile
import unittest
from pathlib import Path

from mega_trading.backtest_examples import run_backtest_examples
from mega_trading.config import BuildConfig
from mega_trading.core.store import LocalObjectStore
from mega_trading.data.ingest_config import IngestPipelineConfig, IngestSourceConfig
from mega_trading.prepare import prepare_numpy_dataset


class BacktestExamplesTests(unittest.TestCase):
    def test_generates_multiple_decoded_backtest_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepare_numpy_dataset(
                IngestPipelineConfig(
                    output_dir=str(root),
                    sources=(IngestSourceConfig(name="fixture"),),
                ),
                BuildConfig(
                    block_size=4,
                    stride=2,
                    min_events_per_ticker=2,
                    validation_fraction=0.2,
                    backtest_fraction=0.3,
                ),
            )

            result = run_backtest_examples(
                LocalObjectStore(root),
                run_id="examples-run",
                mixture_name="public",
                max_sequences=2,
                max_tokens=4,
            )
            payload = LocalObjectStore(root).read_json(result.report_path)

            self.assertEqual(result.report_path, "evals/examples-run/backtest-examples.json")
            self.assertEqual(payload["stage"], "backtest_examples")
            self.assertEqual(payload["run_id"], "examples-run")
            self.assertGreaterEqual(payload["split_counts"]["backtest"], 2)
            self.assertEqual(len(payload["examples"]), 2)
            self.assertLessEqual(len(payload["examples"][0]["tokens"]), 4)
            self.assertIn("token_name", payload["examples"][0]["tokens"][0])
            self.assertIn("price_depth_bps", payload["examples"][0]["tokens"][0])


if __name__ == "__main__":
    unittest.main()
