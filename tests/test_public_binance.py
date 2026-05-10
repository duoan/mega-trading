import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

from mega_trading.data.ingest import BinanceTradesIngestRequest
from mega_trading.config import BuildConfig
from mega_trading.data.ingest_config import IngestPipelineConfig, IngestSourceConfig
from mega_trading.data.public.binance import (
    BINANCE_TRADES_BASE_URL,
    BinanceTradeArchive,
    _configure_certifi_ca_bundle,
    _load_binance_trade_rows,
    _trade_rows_to_order_flow,
    download_binance_trade_archives,
    load_order_flow_from_archives,
)
from mega_trading.prepare import prepare_numpy_dataset


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

    def test_default_download_uses_binance_datatool_for_requested_archives_only(self) -> None:
        request = BinanceTradesIngestRequest(
            symbols=("BTCUSDT",),
            start="2024-01-01T00:00:00Z",
            end="2024-01-01T00:00:03Z",
            frequency="daily",
            process_workers=1,
        )

        def fake_download(requests, **_kwargs):
            for item in requests:
                item.local_path.parent.mkdir(parents=True, exist_ok=True)
                item.local_path.write_bytes(_trades_zip())
            return SimpleNamespace(failed_requests=[])

        available_keys = {"data/spot/daily/trades/BTCUSDT/BTCUSDT-trades-2024-01-01.zip"}
        with tempfile.TemporaryDirectory() as tmp:
            with patch("mega_trading.data.public.binance._list_available_trade_archive_keys", return_value=available_keys):
                with patch("binance_datatool.archive.download_archive_files", side_effect=fake_download) as download:
                    archives = download_binance_trade_archives(request, Path(tmp), BINANCE_TRADES_BASE_URL)
            events = load_order_flow_from_archives(request, archives)

            requested = download.call_args.args[0]
            self.assertEqual(len(requested), 1)
            self.assertIn("BTCUSDT-trades-2024-01-01.zip", requested[0].url)
            self.assertIn("data/spot/daily/trades/BTCUSDT", str(requested[0].local_path))
            self.assertEqual(len(events), 2)

    def test_default_download_skips_missing_symbol_month_archives(self) -> None:
        request = BinanceTradesIngestRequest(
            symbols=("BTCUSDT",),
            start="2024-01-01T00:00:00Z",
            end="2024-01-02T00:00:03Z",
            frequency="daily",
            process_workers=1,
        )
        available_keys = {"data/spot/daily/trades/BTCUSDT/BTCUSDT-trades-2024-01-01.zip"}

        def fake_download(requests, **_kwargs):
            for item in requests:
                item.local_path.parent.mkdir(parents=True, exist_ok=True)
                item.local_path.write_bytes(_trades_zip())
            return SimpleNamespace(failed_requests=[])

        with tempfile.TemporaryDirectory() as tmp:
            with patch("mega_trading.data.public.binance._list_available_trade_archive_keys", return_value=available_keys):
                with patch("binance_datatool.archive.download_archive_files", side_effect=fake_download) as download:
                    archives = download_binance_trade_archives(request, Path(tmp), BINANCE_TRADES_BASE_URL)

            self.assertEqual(len(download.call_args.args[0]), 1)
            self.assertEqual(len(archives), 1)
            self.assertEqual(archives[0].partition, "2024-01-01")

    def test_datatool_download_configures_certifi_ca_bundle(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            _configure_certifi_ca_bundle()

            import os

            self.assertTrue(os.environ["SSL_CERT_FILE"].endswith("cacert.pem"))
            self.assertEqual(os.environ["REQUESTS_CA_BUNDLE"], os.environ["SSL_CERT_FILE"])
            self.assertEqual(os.environ["CURL_CA_BUNDLE"], os.environ["SSL_CERT_FILE"])

    def test_streaming_prepare_writes_partitioned_numpy_without_materializing_downloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_path = root / "raw" / "BTCUSDT-trades-2024-01-01.zip"
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            archive_path.write_bytes(_trades_zip(row_count=12))
            archive = BinanceTradeArchive(
                symbol="BTCUSDT",
                partition="2024-01-01",
                url="https://example.test/BTCUSDT-trades-2024-01-01.zip",
                path=archive_path,
                key="data/spot/daily/trades/BTCUSDT/BTCUSDT-trades-2024-01-01.zip",
            )
            ingest_config = IngestPipelineConfig(
                output_dir=str(root),
                sources=(
                    IngestSourceConfig(
                        name="binance_trades",
                        tickers=("BTCUSDT",),
                        start="2024-01-01T00:00:00Z",
                        end="2024-01-01T00:00:20Z",
                        frequency="daily",
                        process_workers=1,
                    ),
                ),
            )
            build_config = BuildConfig(
                block_size=4,
                stride=2,
                min_events_per_ticker=2,
                numpy_partition_rows=2,
                streaming_prepare=True,
                streaming_tokenizer_sample_events=8,
                streaming_baseline_sample_rows=8,
            )
            with patch("mega_trading.prepare.download_binance_trade_archives", return_value=[archive]):
                result = prepare_numpy_dataset(ingest_config, build_config)

            metadata = json.loads(root.joinpath(result.numpy_metadata_path).read_text(encoding="utf-8"))
            profile = json.loads(root.joinpath(result.profile_path).read_text(encoding="utf-8"))
            self.assertTrue(metadata["partitioned"])
            self.assertGreater(metadata["sequence_count"], 0)
            self.assertTrue(profile["streaming_prepare"])


def _trades_zip(row_count: int = 3) -> bytes:
    payload = io.BytesIO()
    rows = "\n".join(
        f"{index},{42000.0 + index},0.{(index % 9) + 1:02d},{4200.0 + index},"
        f"{1704067200000 + index * 1000},{str(index % 2 != 0).lower()},true"
        for index in range(1, row_count + 1)
    )
    with ZipFile(payload, "w") as archive:
        archive.writestr("BTCUSDT-trades-2024-01.csv", rows + "\n")
    return payload.getvalue()


if __name__ == "__main__":
    unittest.main()
