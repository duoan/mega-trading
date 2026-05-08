import tempfile
import unittest
from pathlib import Path

from marketfm.data.ingest_config import IngestPipelineConfig, load_ingest_config


class IngestConfigTests(unittest.TestCase):
    def test_loads_ingest_pipeline_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "ingest.toml"
            config_path.write_text(
                """
[ingest]
output_dir = ".marketfm/public"
sec_user_agent = "MarketFM test@example.com"

[[ingest.sources]]
name = "sec_companyfacts"
tickers = ["AAPL", "MSFT"]

[[ingest.sources]]
name = "yahoo_prices"
tickers = ["AAPL", "MSFT"]
start = "2024-01-01"
end = "2024-03-31"
""".strip()
                + "\n",
                encoding="utf-8",
            )

            config = load_ingest_config(config_path)

            self.assertEqual(config.output_dir, ".marketfm/public")
            self.assertEqual(config.sec_user_agent, "MarketFM test@example.com")
            self.assertEqual(config.sources[0].name, "sec_companyfacts")
            self.assertEqual(config.sources[1].tickers, ("AAPL", "MSFT"))
            self.assertEqual(config.sources[1].start, "2024-01-01")

    def test_rejects_price_source_without_date_window(self) -> None:
        with self.assertRaises(ValueError):
            IngestPipelineConfig.from_dict(
                {
                    "output_dir": ".marketfm/public",
                    "sources": [{"name": "yahoo_prices", "tickers": ["AAPL"]}],
                }
            )


if __name__ == "__main__":
    unittest.main()
