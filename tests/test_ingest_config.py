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

    def test_checked_in_local_and_modal_ingest_configs_parse(self) -> None:
        binance_local = load_ingest_config(Path("configs/ingest-binance-local.toml"))
        binance_modal_prep = load_ingest_config(Path("configs/ingest-binance-modal-prep.toml"))
        self.assertEqual(binance_local.output_dir, ".mega-trading/binance-local")
        self.assertEqual(binance_local.sources[0].name, "binance_trades")
        self.assertIn("BNBUSDT", binance_local.sources[0].tickers)
        self.assertIn("BTCUSDT", binance_local.sources[0].tickers)
        self.assertEqual(binance_local.sources[0].frequency, "daily")
        self.assertEqual(binance_local.sources[0].download_workers, 8)
        self.assertEqual(binance_local.sources[0].process_workers, 0)
        self.assertEqual(binance_modal_prep.output_dir, ".mega-trading/binance-modal")
        self.assertGreaterEqual(len(binance_modal_prep.sources[0].tickers), 100)
        self.assertEqual(binance_modal_prep.sources[0].download_workers, 16)
        self.assertEqual(binance_modal_prep.sources[0].process_workers, 0)


if __name__ == "__main__":
    unittest.main()
