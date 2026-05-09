import tempfile
import unittest
from pathlib import Path

from mega_trading.data.ingest_config import IngestPipelineConfig, load_ingest_config


class IngestConfigTests(unittest.TestCase):
    def test_loads_ingest_pipeline_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "ingest.toml"
            config_path.write_text(
                """
[ingest]
output_dir = ".mega-trading/public"
sec_user_agent = "Mega-Trading test@example.com"

[[ingest.sources]]
name = "sec_filings"
tickers = ["AAPL", "AMZN"]

[[ingest.sources]]
name = "yahoo_market_data"
tickers = ["AAPL", "AMZN"]
start = "2024-01-01"
end = "2024-03-31"
""".strip()
                + "\n",
                encoding="utf-8",
            )

            config = load_ingest_config(config_path)

            self.assertEqual(config.output_dir, ".mega-trading/public")
            self.assertEqual(config.sec_user_agent, "Mega-Trading test@example.com")
            self.assertEqual(config.sources[0].name, "sec_filings")
            self.assertEqual(config.sources[1].tickers, ("AAPL", "AMZN"))
            self.assertEqual(config.sources[1].start, "2024-01-01")
            self.assertTrue(config.quality_enabled)
            self.assertTrue(config.enrichment_enabled)
            self.assertTrue(config.training_data_enabled)
            self.assertEqual(config.training_mixture_name, "public")

    def test_rejects_market_data_source_without_date_window(self) -> None:
        with self.assertRaises(ValueError):
            IngestPipelineConfig.from_dict(
                {
                    "output_dir": ".mega-trading/public",
                    "sources": [{"name": "yahoo_market_data", "tickers": ["AAPL"]}],
                }
            )

    def test_fixture_source_does_not_require_tickers(self) -> None:
        config = IngestPipelineConfig.from_dict(
            {
                "output_dir": ".mega-trading/demo",
                "sources": [{"name": "fixture"}],
            }
        )

        self.assertEqual(config.sources[0].name, "fixture")
        self.assertEqual(config.sources[0].tickers, ())

    def test_loads_source_tickers_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root.joinpath("sp500.txt").write_text("aapl\nbrk.b\n# comment\nMSFT\n", encoding="utf-8")
            config_path = root / "ingest.toml"
            config_path.write_text(
                """
[ingest]
output_dir = ".mega-trading/public"

[[ingest.sources]]
name = "sec_filings"
tickers = ["amzn"]
ticker_file = "sp500.txt"
""".strip()
                + "\n",
                encoding="utf-8",
            )

            config = load_ingest_config(config_path)

            self.assertEqual(config.sources[0].tickers, ("AAPL", "AMZN", "BRK-B", "MSFT"))

    def test_loads_quality_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "ingest.toml"
            config_path.write_text(
                """
[ingest]
output_dir = ".mega-trading/public"

[ingest.quality]
enabled = true
fail_on_error = true

[ingest.enrichment]
enabled = false

[ingest.training_data]
enabled = false
mixture_name = "disabled"
sequence_length = 16
input_window_observations = 4
horizon_observations = 2
return_threshold = 0.05
workers = 4

[[ingest.sources]]
name = "yahoo_market_data"
tickers = ["AAPL"]
start = "2024-01-01"
end = "2024-01-31"
""".strip()
                + "\n",
                encoding="utf-8",
            )

            config = load_ingest_config(config_path)

            self.assertTrue(config.quality_enabled)
            self.assertTrue(config.quality_fail_on_error)
            self.assertFalse(config.enrichment_enabled)
            self.assertFalse(config.training_data_enabled)
            self.assertEqual(config.training_mixture_name, "disabled")
            self.assertEqual(config.training_sequence_length, 16)
            self.assertEqual(config.training_input_window_observations, 4)
            self.assertEqual(config.training_horizon_observations, 2)
            self.assertEqual(config.training_return_threshold, 0.05)
            self.assertEqual(config.training_workers, 4)


if __name__ == "__main__":
    unittest.main()
