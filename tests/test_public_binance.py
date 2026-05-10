import io
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from mega_trading.data.ingest import BinanceTradesIngestRequest
from mega_trading.data.public.binance import (
    BINANCE_TRADES_BASE_URL,
    _load_binance_trade_rows,
    _trade_rows_to_order_flow,
    download_binance_trade_archives,
    load_order_flow_from_archives,
)


class BinanceTradesTests(unittest.TestCase):
    def test_downloaded_trades_map_to_order_flow_events(self) -> None:
        fetched_urls: list[str] = []

        def fetch_zip(url: str) -> bytes:
            fetched_urls.append(url)
            return _trades_zip()

        raw_rows = _load_binance_trade_rows(
            BinanceTradesIngestRequest(
                symbols=("BTCUSDT",),
                start="2024-01-01T00:00:00Z",
                end="2024-01-01T00:00:03Z",
            ),
            BINANCE_TRADES_BASE_URL,
            fetch_zip,
            )
        events = _trade_rows_to_order_flow(raw_rows)

        self.assertIn("BTCUSDT-trades-2024-01.zip", fetched_urls[0])
        self.assertEqual(len(raw_rows), 3)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].ticker, "BTCUSDT")
        self.assertEqual(events[0].action, "delete")
        self.assertEqual(events[0].side, "buy")
        self.assertGreater(events[0].price_depth_bps, 0.0)
        self.assertGreater(events[0].size, 0.0)

    def test_archives_are_cached_and_processed_from_disk(self) -> None:
        fetched_urls: list[str] = []

        def fetch_zip(url: str) -> bytes:
            fetched_urls.append(url)
            return _trades_zip()

        request = BinanceTradesIngestRequest(
            symbols=("BTCUSDT",),
            start="2024-01-01T00:00:00Z",
            end="2024-01-01T00:00:03Z",
            frequency="daily",
            download_workers=2,
            process_workers=1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            archives = download_binance_trade_archives(request, Path(tmp), BINANCE_TRADES_BASE_URL, fetch_zip)
            events = load_order_flow_from_archives(request, archives)

            self.assertEqual(len(fetched_urls), 1)
            self.assertTrue(archives[0].path.exists())
            self.assertIn("stage=01_raw", str(archives[0].path))
            self.assertEqual(len(events), 2)
            self.assertEqual(events[0].ticker, "BTCUSDT")


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
