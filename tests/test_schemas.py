import unittest

from mega_trading.core.hashing import stable_hash
from mega_trading.core.schemas import Manifest, OrderFlowEventRecord, SchemaValidationError


class SchemaTests(unittest.TestCase):
    def test_order_flow_record_validates_paper_features(self) -> None:
        with self.assertRaises(SchemaValidationError):
            OrderFlowEventRecord(
                event_id="evt-1",
                ticker="ACME",
                timestamp="2024-01-02T14:30:00Z",
                date="2024-01-02",
                action="trade",
                side="buy",
                midprice=100.0,
                relative_price_bps=1.0,
                price_depth_bps=1.0,
                size=100.0,
                interarrival_seconds=1.0,
                provider="fixture",
                source_ids=["raw-1"],
            )

    def test_order_flow_record_accepts_valid_event(self) -> None:
        record = OrderFlowEventRecord(
            event_id="evt-1",
            ticker="ACME",
            timestamp="2024-01-02T14:30:00Z",
            date="2024-01-02",
            action="add",
            side="buy",
            midprice=100.0,
            relative_price_bps=1.0,
            price_depth_bps=1.0,
            size=100.0,
            interarrival_seconds=1.0,
            provider="fixture",
            source_ids=["raw-1"],
        )

        self.assertEqual(record.action, "add")

    def test_manifest_hash_is_deterministic(self) -> None:
        left = Manifest(
            manifest_id="manifest-1",
            artifact_type="events",
            paths=["stage=04_corpus/mixture=demo/events.jsonl"],
            metadata={"b": 2, "a": 1},
        )
        right = Manifest(
            manifest_id="manifest-1",
            artifact_type="events",
            paths=["stage=04_corpus/mixture=demo/events.jsonl"],
            metadata={"a": 1, "b": 2},
        )

        self.assertEqual(left.content_hash(), right.content_hash())
        self.assertEqual(left.content_hash(), stable_hash(right.to_dict()))


if __name__ == "__main__":
    unittest.main()
