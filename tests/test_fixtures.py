import unittest

from mega_trading.data.fixtures import load_fixture_bundle


class FixtureTests(unittest.TestCase):
    def test_fixture_loader_returns_expected_counts(self) -> None:
        bundle = load_fixture_bundle()

        self.assertEqual(len(bundle.entities), 2)
        self.assertEqual(len(bundle.documents), 2)
        self.assertEqual(len(bundle.sec_filings), 4)
        self.assertEqual(len(bundle.market_data), 8)

    def test_fixture_timestamps_are_deterministic(self) -> None:
        left = load_fixture_bundle()
        right = load_fixture_bundle()

        self.assertEqual(left.documents[0].as_of_time, right.documents[0].as_of_time)
        self.assertEqual(left.sec_filings[0].as_of_time, "2023-02-15T16:30:00Z")

    def test_bad_fixtures_are_classified_by_reason(self) -> None:
        bundle = load_fixture_bundle()

        reasons = {record["reason"] for record in bundle.bad_records}

        self.assertEqual(reasons, {"stale_feed", "future_leakage", "missing_entity"})


if __name__ == "__main__":
    unittest.main()
