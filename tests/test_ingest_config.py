import tempfile
import unittest
from pathlib import Path

from mega_trading.data.ingest_config import IngestPipelineConfig, load_ingest_config


class IngestConfigTests(unittest.TestCase):
    def test_loads_binance_order_flow_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "ingest.toml"
            config_path.write_text(
                """
[ingest]
output_dir = ".mega-trading/binance"

[[ingest.sources]]
name = "binance_trades"
tickers = ["BTCUSDT", "ETHUSDT"]
start = "2024-01-01T00:00:00Z"
end = "2024-01-01T23:59:59Z"
frequency = "daily"
""".strip()
                + "\n",
                encoding="utf-8",
            )

            config = load_ingest_config(config_path)

            self.assertEqual(config.output_dir, ".mega-trading/binance")
            self.assertEqual(config.sources[0].name, "binance_trades")
            self.assertEqual(config.sources[0].tickers, ("BTCUSDT", "ETHUSDT"))

    def test_rejects_non_paper_sources(self) -> None:
        with self.assertRaises(ValueError):
            IngestPipelineConfig.from_dict(
                {
                    "output_dir": ".mega-trading/public",
                    "sources": [{"name": "fundamentals", "tickers": ["AAPL"]}],
                }
            )

    def test_binance_source_requires_symbols_and_date_window(self) -> None:
        with self.assertRaises(ValueError):
            IngestPipelineConfig.from_dict(
                {
                    "output_dir": ".mega-trading/binance",
                    "sources": [{"name": "binance_trades", "tickers": ["BTCUSDT"]}],
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

    def test_checked_in_environment_ingest_configs_parse(self) -> None:
        mac = load_ingest_config(Path("configs/ingest-mac.toml"))
        rtx = load_ingest_config(Path("configs/ingest-rtx.toml"))
        modal = load_ingest_config(Path("configs/ingest-modal.toml"))
        self.assertEqual(mac.output_dir, ".mega-trading/data")
        self.assertEqual(mac.raw_dir, ".mega-trading/raw")
        self.assertEqual(mac.sources[0].name, "binance_trades")
        self.assertIn("BNBUSDT", mac.sources[0].tickers)
        self.assertIn("BTCUSDT", mac.sources[0].tickers)
        self.assertEqual(mac.sources[0].frequency, "daily")
        self.assertEqual(mac.sources[0].download_workers, 8)
        self.assertEqual(mac.sources[0].process_workers, 0)
        self.assertEqual(rtx.output_dir, ".mega-trading/data")
        self.assertEqual(rtx.raw_dir, ".mega-trading/raw")
        self.assertGreaterEqual(len(rtx.sources[0].tickers), 30)
        self.assertLess(len(rtx.sources[0].tickers), len(modal.sources[0].tickers))
        self.assertIn("BTCUSDT", rtx.sources[0].tickers)
        self.assertIn("ETHUSDT", rtx.sources[0].tickers)
        self.assertEqual(rtx.sources[0].download_workers, 16)
        self.assertEqual(rtx.sources[0].process_workers, 0)
        self.assertEqual(modal.output_dir, ".mega-trading/data")
        self.assertEqual(modal.raw_dir, ".mega-trading/raw")
        self.assertGreaterEqual(len(modal.sources[0].tickers), 100)
        self.assertEqual(modal.sources[0].download_workers, 16)
        self.assertEqual(modal.sources[0].process_workers, 0)


if __name__ == "__main__":
    unittest.main()
