import tempfile
import unittest
from pathlib import Path

from mega_trading.core.store import LocalObjectStore
from mega_trading.data.ingest import OhlcvIngestRequest
from mega_trading.data.public.market import HuggingFaceOhlcvIngestor


class HuggingFaceOhlcvTests(unittest.TestCase):
    def test_ingestor_filters_huggingface_dataset_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            HuggingFaceOhlcvIngestor(store, load_dataset_fn=_load_dataset_stub).ingest(
                OhlcvIngestRequest(
                    tickers=("AAPL",),
                    start="2024-01-02T14:30:00Z",
                    end="2024-01-02T14:32:00Z",
                )
            )
            rows = store.read_jsonl("stage=01_raw/source=hf_ohlcv_1m/ohlcv.jsonl")

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["ticker"], "AAPL")
        self.assertEqual(rows[0]["timestamp"], "2024-01-02T14:30:00Z")

    def test_ingestor_writes_paper_order_flow_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            result = HuggingFaceOhlcvIngestor(store, load_dataset_fn=_load_dataset_stub).ingest(
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


def _load_dataset_stub(name: str, data_files: str, split: str, streaming: bool):
    if name != "mito0o852/OHLCV-1m":
        raise AssertionError(name)
    if data_files != "data/ohlcv_2024-01.parquet":
        raise AssertionError(data_files)
    if split != "train" or not streaming:
        raise AssertionError((split, streaming))
    return [
        {
            "timestamp": "2024-01-02T14:30:00Z",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000.0,
            "ticker": "AAPL",
        },
        {
            "timestamp": "2024-01-02T14:31:00Z",
            "open": 100.5,
            "high": 101.2,
            "low": 100.1,
            "close": 100.2,
            "volume": 1200.0,
            "ticker": "AAPL",
        },
        {
            "timestamp": "2024-01-02T14:32:00Z",
            "open": 100.2,
            "high": 100.8,
            "low": 99.8,
            "close": 100.7,
            "volume": 900.0,
            "ticker": "AAPL",
        },
        {
            "timestamp": "2024-01-02T14:30:00Z",
            "open": 200.0,
            "high": 201.0,
            "low": 199.0,
            "close": 200.5,
            "volume": 500.0,
            "ticker": "MSFT",
        },
    ]


if __name__ == "__main__":
    unittest.main()
