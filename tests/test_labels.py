import unittest

from mega_trading.data.labels import LabelConfig, ForwardLabelGenerator


class ForwardLabelGeneratorTests(unittest.TestCase):
    def test_generates_forward_return_labels_after_as_of_date(self) -> None:
        market_data = [
            _market_data("ACME", "2024-01-01", 100.0),
            _market_data("ACME", "2024-01-02", 110.0),
            _market_data("ACME", "2024-01-03", 120.0),
            _market_data("ACME", "2024-01-04", 130.0),
        ]

        labels = ForwardLabelGenerator(
            LabelConfig(input_window_observations=2, horizon_observations=2, return_threshold=0.05)
        ).generate(market_data)

        self.assertEqual(len(labels), 1)
        label = labels[0]
        self.assertEqual(label["ticker"], "ACME")
        self.assertEqual(label["as_of_date"], "2024-01-02")
        self.assertEqual(label["label_start_date"], "2024-01-03")
        self.assertEqual(label["label_end_date"], "2024-01-04")
        self.assertEqual(label["forward_return_bucket"], "outperform")
        self.assertAlmostEqual(label["forward_return"], (130.0 / 110.0) - 1.0)

    def test_generates_underperform_bucket_for_negative_forward_return(self) -> None:
        market_data = [
            _market_data("NOVA", "2024-01-01", 100.0),
            _market_data("NOVA", "2024-01-02", 100.0),
            _market_data("NOVA", "2024-01-03", 94.0),
            _market_data("NOVA", "2024-01-04", 90.0),
        ]

        labels = ForwardLabelGenerator(
            LabelConfig(input_window_observations=2, horizon_observations=2, return_threshold=0.05)
        ).generate(market_data)

        self.assertEqual(labels[0]["forward_return_bucket"], "underperform")
        self.assertEqual(labels[0]["risk_bucket"], "high")

    def test_skips_examples_without_full_input_or_label_window(self) -> None:
        market_data = [
            _market_data("ACME", "2024-01-01", 100.0),
            _market_data("ACME", "2024-01-02", 101.0),
            _market_data("ACME", "2024-01-03", 102.0),
        ]

        labels = ForwardLabelGenerator(
            LabelConfig(input_window_observations=3, horizon_observations=2, return_threshold=0.05)
        ).generate(market_data)

        self.assertEqual(labels, [])


def _market_data(ticker: str, date: str, adjusted_close: float) -> dict[str, object]:
    return {
        "market_data_id": f"market_data-{ticker}-{date}",
        "ticker": ticker,
        "date": date,
        "adjusted_close": adjusted_close,
        "provider": "fixture",
        "source_ids": [f"fixture:{ticker}:{date}"],
    }


if __name__ == "__main__":
    unittest.main()
