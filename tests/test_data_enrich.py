import tempfile
import unittest
from pathlib import Path

from mega_trading.core.store import LocalObjectStore
from mega_trading.data.enrich import DataEnricher


class DataEnrichmentTests(unittest.TestCase):
    def test_enricher_writes_company_snapshot_from_sec_and_market_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            store.write_jsonl(
                "stage=02_normalized/family=entities/source=sec.jsonl",
                [{"entity_id": "sec-1", "ticker": "AAPL", "company_name": "Apple Inc.", "source_ids": ["sec:1"]}],
            )
            store.write_jsonl(
                "stage=02_normalized/family=sec_filings/source=sec.jsonl",
                [
                    {
                        "sec_filing_id": "f1",
                        "entity_id": "sec-1",
                        "ticker": "AAPL",
                        "concept": "Revenue",
                        "value": 100.0,
                        "unit": "USD",
                        "period_end": "2023-12-31",
                        "accepted_at": "2024-02-01T00:00:00Z",
                        "as_of_time": "2024-02-01T00:00:00Z",
                        "source_ids": ["sec:f1"],
                    }
                ],
            )
            store.write_jsonl(
                "stage=02_normalized/family=market_data/source=yahoo.jsonl",
                [
                    {
                        "market_data_id": "p1",
                        "ticker": "AAPL",
                        "date": "2024-01-02",
                        "adjusted_close": 185.0,
                        "provider": "yahoo",
                        "source_ids": ["p1"],
                    },
                    {
                        "market_data_id": "p2",
                        "ticker": "AAPL",
                        "date": "2024-01-03",
                        "adjusted_close": 186.0,
                        "provider": "yahoo",
                        "source_ids": ["p2"],
                    },
                ],
            )

            result = DataEnricher(store).run(run_id="unit")
            snapshots = store.read_jsonl(result.snapshot_path)

            self.assertEqual(result.snapshots, 1)
            self.assertEqual(snapshots[0]["ticker"], "AAPL")
            self.assertEqual(snapshots[0]["market_data_observations"], 2)
            self.assertEqual(snapshots[0]["latest_sec_filing_as_of_time"], "2024-02-01T00:00:00Z")


if __name__ == "__main__":
    unittest.main()
