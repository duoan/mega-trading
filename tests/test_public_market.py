import tempfile
import unittest
from pathlib import Path

import pyarrow as pa

from mega_trading.core.store import LocalObjectStore
from mega_trading.data.ingest import OhlcvIngestRequest
from mega_trading.data.public.market import HuggingFaceOhlcvClient, HuggingFaceOhlcvIngestor


class HuggingFaceOhlcvTests(unittest.TestCase):
    def test_client_filters_minute_parquet_rows(self) -> None:
        def fetch_table(url: str) -> pa.Table:
            self.assertIn("ohlcv_2024-01.parquet", url)
            return _minute_table()

        rows = HuggingFaceOhlcvClient(fetch_table=fetch_table).minute(
            ("AAPL",),
            "2024-01-02T14:30:00Z",
            "2024-01-02T14:32:00Z",
        )

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["ticker"], "AAPL")
        self.assertEqual(rows[0]["timestamp"], "2024-01-02T14:30:00Z")

    def test_ingestor_writes_paper_order_flow_artifacts(self) -> None:
        def fetch_table(_url: str) -> pa.Table:
            return _minute_table()

        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            result = HuggingFaceOhlcvIngestor(store, HuggingFaceOhlcvClient(fetch_table=fetch_table)).ingest(
                OhlcvIngestRequest(
                    tickers=("AAPL",),
                    start="2024-01-02T14:30:00Z",
                    end="2024-01-02T14:32:00Z",
                )
            )
            events = store.read_jsonl("stage=02_normalized/family=order_flow/source=hf_ohlcv_1m.jsonl")
            manifest = store.read_manifest(result.normalization_manifest_path)

            self.assertEqual(len(events), 2)
            self.assertEqual(events[0]["action"], "add")
            self.assertIn(events[0]["side"], {"buy", "sell"})
            self.assertAlmostEqual(events[0]["midprice"], 100.65)
            self.assertAlmostEqual(events[0]["size"], 1.2)
            self.assertLess(events[0]["relative_price_bps"], 0.0)
            self.assertGreaterEqual(events[0]["price_depth_bps"], 0.0)
            self.assertEqual(manifest.metadata["feature_contract"], "paper-order-flow-v1")


def _minute_table() -> pa.Table:
    return pa.table(
        {
            "timestamp": [
                "2024-01-02T14:30:00Z",
                "2024-01-02T14:31:00Z",
                "2024-01-02T14:32:00Z",
                "2024-01-02T14:30:00Z",
            ],
            "open": [100.0, 100.5, 100.2, 200.0],
            "high": [101.0, 101.2, 100.8, 201.0],
            "low": [99.0, 100.1, 99.8, 199.0],
            "close": [100.5, 100.2, 100.7, 200.5],
            "volume": [1000.0, 1200.0, 900.0, 500.0],
            "ticker": ["AAPL", "AAPL", "AAPL", "MSFT"],
        }
    )


if __name__ == "__main__":
    unittest.main()
