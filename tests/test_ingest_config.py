import tempfile
import unittest
from pathlib import Path

from mega_trading.data.ingest_config import IngestPipelineConfig, load_ingest_config


class IngestConfigTests(unittest.TestCase):
    def test_loads_hf_order_flow_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "ingest.toml"
            config_path.write_text(
                """
[ingest]
output_dir = ".mega-trading/hf-1m"

[[ingest.sources]]
name = "hf_ohlcv_1m"
tickers = ["AAPL", "MSFT"]
start = "2024-01-02T14:30:00Z"
end = "2024-01-02T21:00:00Z"
""".strip()
                + "\n",
                encoding="utf-8",
            )

            config = load_ingest_config(config_path)

            self.assertEqual(config.output_dir, ".mega-trading/hf-1m")
            self.assertEqual(config.sources[0].name, "hf_ohlcv_1m")
            self.assertEqual(config.sources[0].tickers, ("AAPL", "MSFT"))
            self.assertTrue(config.quality_enabled)

    def test_rejects_non_paper_sources(self) -> None:
        with self.assertRaises(ValueError):
            IngestPipelineConfig.from_dict(
                {
                    "output_dir": ".mega-trading/public",
                    "sources": [{"name": "fundamentals", "tickers": ["AAPL"]}],
                }
            )

    def test_hf_source_requires_date_window(self) -> None:
        with self.assertRaises(ValueError):
            IngestPipelineConfig.from_dict(
                {
                    "output_dir": ".mega-trading/hf-1m",
                    "sources": [{"name": "hf_ohlcv_1m", "tickers": ["AAPL"]}],
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


if __name__ == "__main__":
    unittest.main()
