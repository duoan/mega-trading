import unittest

from mega_trading.data.fixtures import load_fixture_bundle


class FixtureTests(unittest.TestCase):
    def test_fixture_loader_returns_order_flow(self) -> None:
        bundle = load_fixture_bundle()

        self.assertEqual(len(bundle.order_flow), 24)
        self.assertEqual(bundle.order_flow[0].action, "delete")
        self.assertEqual(bundle.order_flow[0].interarrival_seconds, 60.0)

    def test_fixture_timestamps_are_deterministic(self) -> None:
        left = load_fixture_bundle()
        right = load_fixture_bundle()

        self.assertEqual(left.order_flow[0].timestamp, right.order_flow[0].timestamp)

    def test_bad_fixtures_are_classified_by_reason(self) -> None:
        bundle = load_fixture_bundle()

        reasons = {record["reason"] for record in bundle.bad_records}

        self.assertEqual(reasons, {"invalid_event"})


if __name__ == "__main__":
    unittest.main()
