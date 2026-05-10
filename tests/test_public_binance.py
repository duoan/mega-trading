import io
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from mega_trading.core.store import LocalObjectStore
from mega_trading.data.ingest import BinanceTradesIngestRequest
from mega_trading.data.public.binance import BinanceTradesIngestor


class BinanceTradesTests(unittest.TestCase):
    def test_ingestor_writes_trade_level_order_flow_events(self) -> None:
        fetched_urls: list[str] = []

        def fetch_zip(url: str) -> bytes:
            fetched_urls.append(url)
            return _trades_zip()

        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            result = BinanceTradesIngestor(store, fetch_zip=fetch_zip).ingest(
                BinanceTradesIngestRequest(
                    symbols=("BTCUSDT",),
                    start="2024-01-01T00:00:00Z",
                    end="2024-01-01T00:00:03Z",
                )
            )
            raw_rows = store.read_jsonl("stage=01_raw/source=binance_trades/trades.jsonl")
            events = store.read_jsonl("stage=02_normalized/family=order_flow/source=binance_trades.jsonl")
            manifest = store.read_manifest(result.normalization_manifest_path)

        self.assertIn("BTCUSDT-trades-2024-01.zip", fetched_urls[0])
        self.assertEqual(len(raw_rows), 3)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["ticker"], "BTCUSDT")
        self.assertEqual(events[0]["action"], "delete")
        self.assertEqual(events[0]["side"], "buy")
        self.assertGreater(events[0]["price_depth_bps"], 0.0)
        self.assertGreater(events[0]["size"], 0.0)
        self.assertEqual(manifest.metadata["source"], "binance_trades")


def _trades_zip() -> bytes:
    payload = io.BytesIO()
    rows = "\n".join(
        [
            "1,42000.0,0.10,4200.0,1704067200000,true,true",
            "2,42010.0,0.20,8402.0,1704067201000,false,true",
            "3,42005.0,0.05,2100.25,1704067202000,true,true",
        ]
    )
    with ZipFile(payload, "w") as archive:
        archive.writestr("BTCUSDT-trades-2024-01.csv", rows + "\n")
    return payload.getvalue()


if __name__ == "__main__":
    unittest.main()
