import tempfile
import unittest
from pathlib import Path

from mega_trading.core.schemas import Manifest
from mega_trading.core.store import LocalObjectStore
from mega_trading.data.quality import DataQualityChecker


class DataQualityTests(unittest.TestCase):
    def test_quality_checker_writes_readiness_report_for_valid_order_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            _write_order_flow(store, [{"event_id": "evt-1"}])
            manifest_path = _write_manifest(store)

            result = DataQualityChecker(store).run([manifest_path], run_id="unit")
            report = store.read_json(result.report_path)

            self.assertTrue(result.passed)
            self.assertEqual(report["total_records"], 1)
            self.assertEqual(report["quarantined_records"], 0)
            self.assertTrue(store.read_jsonl(result.metrics_path))

    def test_quality_checker_quarantines_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            _write_order_flow(store, [{"event_id": "evt-1"}, {"event_id": "evt-1"}])
            manifest_path = _write_manifest(store)

            result = DataQualityChecker(store).run([manifest_path], run_id="unit")
            quarantined = store.read_jsonl(result.quarantine_path)

            self.assertFalse(result.passed)
            self.assertEqual(quarantined[0]["reason"], "duplicate_id")

    def test_quality_checker_fails_on_bad_order_flow_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            _write_order_flow(store, [{"event_id": "evt-1", "size": 0.0}])
            manifest_path = _write_manifest(store)

            result = DataQualityChecker(store).run([manifest_path], run_id="unit")

            self.assertFalse(result.passed)
            self.assertEqual(store.read_json(result.report_path)["issues_by_reason"], {"invalid_size": 1})


def _write_order_flow(store: LocalObjectStore, overrides: list[dict[str, object]]) -> None:
    rows = []
    for index, override in enumerate(overrides):
        row = {
            "event_id": f"evt-{index}",
            "ticker": "AAPL",
            "timestamp": f"2024-01-02T14:{30 + index:02d}:00Z",
            "date": "2024-01-02",
            "action": "add",
            "side": "buy",
            "midprice": 100.0,
            "relative_price_bps": 1.0,
            "price_depth_bps": 1.0,
            "size": 100.0,
            "interarrival_seconds": 60.0,
            "provider": "fixture",
            "source_ids": [f"raw-{index}"],
        }
        row.update(override)
        rows.append(row)
    store.write_jsonl("stage=02_normalized/family=order_flow/source=fixture.jsonl", rows)


def _write_manifest(store: LocalObjectStore) -> str:
    manifest_path = "manifests/normalization/fixture-order-flow-normalized.json"
    store.write_manifest(
        manifest_path,
        Manifest(
            manifest_id="fixture-order-flow-normalized",
            artifact_type="normalized",
            paths=["stage=02_normalized/family=order_flow/source=fixture.jsonl"],
        ),
    )
    return manifest_path


if __name__ == "__main__":
    unittest.main()
