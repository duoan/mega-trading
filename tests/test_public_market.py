import tempfile
import unittest
from pathlib import Path

from mega_trading.core.store import LocalObjectStore
from mega_trading.data.ingest import MarketDataIngestRequest
from mega_trading.data.public.market import StooqClient, StooqMarketDataIngestor, YahooChartClient, YahooMarketDataIngestor


class StooqMarketDataTests(unittest.TestCase):
    def test_stooq_client_parses_daily_csv(self) -> None:
        def fetch_text(url: str) -> str:
            self.assertIn("aapl.us", url)
            return "Date,Open,High,Low,Close,Volume\n2023-01-03,100,110,90,105,12345\n"

        client = StooqClient(fetch_text=fetch_text)

        rows = client.daily("AAPL", "2023-01-01", "2023-01-31")

        self.assertEqual(rows[0]["date"], "2023-01-03")
        self.assertEqual(rows[0]["close"], 105.0)

    def test_stooq_ingestor_writes_market_data_artifacts(self) -> None:
        def fetch_text(_url: str) -> str:
            return "Date,Open,High,Low,Close,Volume\n2023-01-03,100,110,90,105,12345\n"

        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            client = StooqClient(fetch_text=fetch_text)

            result = StooqMarketDataIngestor(store, client).ingest(
                MarketDataIngestRequest(tickers=("AAPL",), start="2023-01-01", end="2023-01-31")
            )
            market_data = store.read_jsonl("stage=02_normalized/family=market_data/source=stooq.jsonl")
            manifest = store.read_manifest(result.normalization_manifest_path)

            self.assertEqual(len(market_data), 1)
            self.assertEqual(market_data[0]["ticker"], "AAPL")
            self.assertEqual(market_data[0]["adjusted_close"], 105.0)
            self.assertEqual(manifest.metadata["source"], "stooq_daily")


class YahooMarketDataTests(unittest.TestCase):
    def test_yahoo_client_parses_chart_payload(self) -> None:
        def fetch_json(url: str) -> dict:
            self.assertIn("query1.finance.yahoo.com", url)
            return {
                "chart": {
                    "result": [
                        {
                            "timestamp": [1672704000],
                            "indicators": {
                                "quote": [
                                    {
                                        "open": [100.0],
                                        "high": [110.0],
                                        "low": [90.0],
                                        "close": [105.0],
                                        "volume": [12345],
                                    }
                                ],
                                "adjclose": [{"adjclose": [104.5]}],
                            },
                        }
                    ],
                    "error": None,
                }
            }

        rows = YahooChartClient(fetch_json=fetch_json).daily("AAPL", "2023-01-01", "2023-01-31")

        self.assertEqual(rows[0]["date"], "2023-01-03")
        self.assertEqual(rows[0]["adjusted_close"], 104.5)

    def test_yahoo_ingestor_writes_market_data_artifacts(self) -> None:
        def fetch_json(_url: str) -> dict:
            return {
                "chart": {
                    "result": [
                        {
                            "timestamp": [1672704000],
                            "indicators": {
                                "quote": [
                                    {
                                        "open": [100.0],
                                        "high": [110.0],
                                        "low": [90.0],
                                        "close": [105.0],
                                        "volume": [12345],
                                    }
                                ],
                                "adjclose": [{"adjclose": [104.5]}],
                            },
                        }
                    ],
                    "error": None,
                }
            }

        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))

            result = YahooMarketDataIngestor(store, YahooChartClient(fetch_json=fetch_json)).ingest(
                MarketDataIngestRequest(tickers=("AAPL",), start="2023-01-01", end="2023-01-31")
            )
            market_data = store.read_jsonl("stage=02_normalized/family=market_data/source=yahoo.jsonl")
            manifest = store.read_manifest(result.normalization_manifest_path)

            self.assertEqual(len(market_data), 1)
            self.assertEqual(market_data[0]["ticker"], "AAPL")
            self.assertEqual(manifest.metadata["source"], "yahoo_chart")


if __name__ == "__main__":
    unittest.main()
