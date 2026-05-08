import tempfile
import unittest
from pathlib import Path

from marketfm.core.schemas import Manifest
from marketfm.core.store import LocalObjectStore
from marketfm.data.quality import DataQualityChecker


class DataQualityTests(unittest.TestCase):
    def test_quality_checker_writes_readiness_report_for_valid_silver_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            store.write_jsonl(
                "silver/prices/yahoo.jsonl",
                [
                    {
                        "price_id": "yahoo-AAPL-2024-01-02",
                        "ticker": "AAPL",
                        "date": "2024-01-02",
                        "adjusted_close": 185.0,
                        "provider": "yahoo",
                        "source_ids": ["yahoo-AAPL-2024-01-02"],
                    }
                ],
            )
            manifest_path = "manifests/normalization/yahoo-daily-normalized.json"
            store.write_manifest(
                manifest_path,
                Manifest(
                    manifest_id="yahoo-daily-normalized",
                    artifact_type="silver",
                    paths=["silver/prices/yahoo.jsonl"],
                ),
            )

            result = DataQualityChecker(store).run([manifest_path], run_id="unit")
            report = store.read_json(result.report_path)

            self.assertTrue(result.passed)
            self.assertEqual(report["total_records"], 1)
            self.assertEqual(report["quarantined_records"], 0)
            self.assertTrue(store.read_jsonl(result.metrics_path))

    def test_quality_checker_quarantines_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            store.write_jsonl(
                "silver/entities/sec.jsonl",
                [
                    {"entity_id": "sec-1", "ticker": "AAPL", "company_name": "Apple Inc.", "source_ids": ["one"]},
                    {"entity_id": "sec-1", "ticker": "AAPL", "company_name": "Apple Inc.", "source_ids": ["two"]},
                ],
            )
            manifest_path = "manifests/normalization/sec-companyfacts-normalized.json"
            store.write_manifest(
                manifest_path,
                Manifest(
                    manifest_id="sec-companyfacts-normalized",
                    artifact_type="silver",
                    paths=["silver/entities/sec.jsonl"],
                ),
            )

            result = DataQualityChecker(store).run([manifest_path], run_id="unit")
            quarantined = store.read_jsonl(result.quarantine_path)

            self.assertFalse(result.passed)
            self.assertEqual(quarantined[0]["reason"], "duplicate_id")

    def test_quality_checker_uses_family_specific_record_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            store.write_jsonl(
                "silver/fundamentals/sec.jsonl",
                [
                    {
                        "fundamental_id": "f1",
                        "entity_id": "sec-1",
                        "ticker": "AAPL",
                        "concept": "Revenue",
                        "value": 100.0,
                        "unit": "USD",
                        "period_end": "2024-01-31",
                        "accepted_at": "2024-02-01T00:00:00Z",
                        "as_of_time": "2024-02-01T00:00:00Z",
                        "source_ids": ["f1"],
                    },
                    {
                        "fundamental_id": "f2",
                        "entity_id": "sec-1",
                        "ticker": "AAPL",
                        "concept": "Revenue",
                        "value": 200.0,
                        "unit": "USD",
                        "period_end": "2024-02-29",
                        "accepted_at": "2024-03-01T00:00:00Z",
                        "as_of_time": "2024-03-01T00:00:00Z",
                        "source_ids": ["f2"],
                    },
                ],
            )
            manifest_path = "manifests/normalization/sec-companyfacts-normalized.json"
            store.write_manifest(
                manifest_path,
                Manifest(
                    manifest_id="sec-companyfacts-normalized",
                    artifact_type="silver",
                    paths=["silver/fundamentals/sec.jsonl"],
                ),
            )

            result = DataQualityChecker(store).run([manifest_path], run_id="unit")

            self.assertTrue(result.passed)

    def test_quality_checker_fails_on_bad_price_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            store.write_jsonl(
                "silver/prices/yahoo.jsonl",
                [
                    {
                        "price_id": "bad-price",
                        "ticker": "AAPL",
                        "date": "2024-01-02",
                        "adjusted_close": -1.0,
                        "provider": "yahoo",
                        "source_ids": ["bad-price"],
                    }
                ],
            )
            manifest_path = "manifests/normalization/yahoo-daily-normalized.json"
            store.write_manifest(
                manifest_path,
                Manifest(
                    manifest_id="yahoo-daily-normalized",
                    artifact_type="silver",
                    paths=["silver/prices/yahoo.jsonl"],
                ),
            )

            result = DataQualityChecker(store).run([manifest_path], run_id="unit")

            self.assertFalse(result.passed)
            self.assertEqual(store.read_json(result.report_path)["issues_by_reason"], {"invalid_price": 1})


if __name__ == "__main__":
    unittest.main()
