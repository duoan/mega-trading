import tempfile
import unittest
from pathlib import Path

from marketfm.core.store import LocalObjectStore
from marketfm.data.ingest import TickerIngestRequest
from marketfm.data.public.sec import SecClient, SecCompanyFactsIngestor


class SecClientTests(unittest.TestCase):
    def test_sec_client_resolves_ticker_and_companyfacts_url(self) -> None:
        calls: list[str] = []

        def fetch_json(url: str, _user_agent: str) -> dict:
            calls.append(url)
            if url.endswith("/company_tickers.json"):
                return {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
            return {
                "cik": 320193,
                "entityName": "Apple Inc.",
                "facts": {
                    "us-gaap": {
                        "Revenues": {
                            "label": "Revenue",
                            "units": {
                                "USD": [
                                    {
                                        "val": 383285000000,
                                        "end": "2023-09-30",
                                        "filed": "2023-11-03",
                                        "form": "10-K",
                                        "fy": 2023,
                                        "fp": "FY",
                                    }
                                ]
                            },
                        }
                    }
                },
            }

        client = SecClient(user_agent="MarketFM test@example.com", fetch_json=fetch_json)

        entity = client.resolve_ticker("aapl")
        facts = client.company_facts("AAPL")

        self.assertEqual(entity["cik"], "0000320193")
        self.assertEqual(facts["entityName"], "Apple Inc.")
        self.assertTrue(any("companyfacts/CIK0000320193.json" in url for url in calls))

    def test_sec_companyfacts_ingestor_writes_real_contract_artifacts(self) -> None:
        def fetch_json(url: str, _user_agent: str) -> dict:
            if url.endswith("/company_tickers.json"):
                return {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
            return {
                "cik": 320193,
                "entityName": "Apple Inc.",
                "facts": {
                    "us-gaap": {
                        "Revenues": {
                            "label": "Revenue",
                            "units": {
                                "USD": [
                                    {
                                        "val": 383285000000,
                                        "end": "2023-09-30",
                                        "filed": "2023-11-03",
                                        "form": "10-K",
                                    }
                                ]
                            },
                        },
                        "NetIncomeLoss": {
                            "label": "Net Income",
                            "units": {
                                "USD": [
                                    {
                                        "val": 96995000000,
                                        "end": "2023-09-30",
                                        "filed": "2023-11-03",
                                        "form": "10-K",
                                    }
                                ]
                            },
                        },
                    }
                },
            }

        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            client = SecClient(user_agent="MarketFM test@example.com", fetch_json=fetch_json)

            result = SecCompanyFactsIngestor(store, client).ingest(TickerIngestRequest(tickers=("AAPL",)))

            entities = store.read_jsonl("silver/entities/sec.jsonl")
            fundamentals = store.read_jsonl("silver/fundamentals/sec.jsonl")
            manifest = store.read_manifest(result.normalization_manifest_path)

            self.assertEqual(entities[0]["ticker"], "AAPL")
            self.assertEqual(len(fundamentals), 2)
            self.assertEqual(fundamentals[0]["as_of_time"], "2023-11-03T00:00:00Z")
            self.assertEqual(manifest.metadata["source"], "sec_companyfacts")


if __name__ == "__main__":
    unittest.main()
