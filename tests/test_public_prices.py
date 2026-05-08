import tempfile
import unittest
from pathlib import Path

from marketfm.public.prices import StooqClient, StooqPriceIngestor
from marketfm.store import LocalObjectStore


class StooqPriceTests(unittest.TestCase):
    def test_stooq_client_parses_daily_csv(self) -> None:
        def fetch_text(url: str) -> str:
            self.assertIn("aapl.us", url)
            return "Date,Open,High,Low,Close,Volume\n2023-01-03,100,110,90,105,12345\n"

        client = StooqClient(fetch_text=fetch_text)

        rows = client.daily("AAPL", "2023-01-01", "2023-01-31")

        self.assertEqual(rows[0]["date"], "2023-01-03")
        self.assertEqual(rows[0]["close"], 105.0)

    def test_stooq_ingestor_writes_price_artifacts(self) -> None:
        def fetch_text(_url: str) -> str:
            return "Date,Open,High,Low,Close,Volume\n2023-01-03,100,110,90,105,12345\n"

        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            client = StooqClient(fetch_text=fetch_text)

            result = StooqPriceIngestor(store, client).ingest(["AAPL"], "2023-01-01", "2023-01-31")
            prices = store.read_jsonl("silver/prices/stooq.jsonl")
            manifest = store.read_manifest(result.normalization_manifest_path)

            self.assertEqual(len(prices), 1)
            self.assertEqual(prices[0]["ticker"], "AAPL")
            self.assertEqual(prices[0]["adjusted_close"], 105.0)
            self.assertEqual(manifest.metadata["source"], "stooq_daily")


if __name__ == "__main__":
    unittest.main()
