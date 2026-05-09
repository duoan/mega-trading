import tempfile
import unittest
from pathlib import Path

from mega_trading.core.store import LocalObjectStore
from mega_trading.data.labels import LabelConfig
from mega_trading.data.samples import MultiStreamSampleBuilder


class MultiStreamSampleBuilderTests(unittest.TestCase):
    def test_builds_samples_with_past_inputs_and_future_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = _prepared_store(tmp)

            result = MultiStreamSampleBuilder(
                store,
                LabelConfig(input_window_observations=2, horizon_observations=2, return_threshold=0.05),
            ).build(mixture_name="public")
            samples = store.read_jsonl("stage=04_corpus/mixture=public/samples.jsonl")
            manifest = store.read_manifest(result.manifest_path)

            self.assertEqual(result.samples, 1)
            self.assertEqual(len(samples), 1)
            sample = samples[0]
            self.assertEqual(sample["ticker"], "ACME")
            self.assertEqual(sample["as_of_time"], "2024-01-02T00:00:00Z")
            self.assertEqual([row["date"] for row in sample["market_data_window"]], ["2024-01-01", "2024-01-02"])
            self.assertEqual(sample["labels"]["label_start_date"], "2024-01-03")
            self.assertEqual(sample["labels"]["forward_return_bucket"], "outperform")
            self.assertEqual([row["event_id"] for row in sample["news_window"]], ["news-visible-ACME"])
            self.assertEqual([row["sec_filing_id"] for row in sample["sec_filing_window"]], ["fact-visible-ACME"])
            self.assertEqual([row["earnings_id"] for row in sample["earnings_window"]], ["earnings-visible-ACME"])
            self.assertEqual([row["macro_id"] for row in sample["macro_window"]], ["macro-visible"])
            self.assertNotIn("news-future-source:ACME", sample["source_ids"])
            self.assertNotIn("fact-future", sample["source_ids"])
            self.assertNotIn("earnings-future-source:ACME", sample["source_ids"])
            self.assertNotIn("macro-future-source", sample["source_ids"])
            self.assertIn("stage=04_corpus/mixture=public/samples.jsonl", manifest.paths)

    def test_requires_readiness_report_to_be_training_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            store.write_json("reports/data-readiness.json", {"training_ready": False})

            with self.assertRaises(ValueError):
                MultiStreamSampleBuilder(store).build(mixture_name="public")

    def test_parallel_sample_builder_preserves_ticker_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = _prepared_store(tmp, tickers=("ACME", "BETA"))

            result = MultiStreamSampleBuilder(
                store,
                LabelConfig(input_window_observations=2, horizon_observations=2, return_threshold=0.05),
                num_workers=2,
            ).build(mixture_name="public")
            manifest = store.read_manifest(result.manifest_path)
            samples = [sample for path in manifest.paths for sample in store.read_jsonl(path)]

            self.assertEqual(result.samples, 2)
            self.assertEqual({sample["ticker"] for sample in samples}, {"ACME", "BETA"})
            self.assertEqual(manifest.metadata["workers"], "2")
            self.assertEqual(manifest.metadata["partitioned"], "true")


def _prepared_store(tmp: str, tickers: tuple[str, ...] = ("ACME",)) -> LocalObjectStore:
    store = LocalObjectStore(Path(tmp))
    store.write_json("reports/data-readiness.json", {"training_ready": True, "quality_score": 1.0})
    store.write_jsonl(
        "stage=02_normalized/family=entities/source=sec.jsonl",
        [
            {
                "entity_id": f"entity-{ticker.lower()}",
                "ticker": ticker,
                "company_name": f"{ticker} Inc.",
                "source_ids": [f"entity-source:{ticker}"],
            }
            for ticker in tickers
        ],
    )
    store.write_jsonl(
        "stage=02_normalized/family=sec_filings/source=sec.jsonl",
        [
            fact
            for ticker in tickers
            for fact in (
                {
                    "sec_filing_id": f"fact-visible-{ticker}",
                    "entity_id": f"entity-{ticker.lower()}",
                    "ticker": ticker,
                    "concept": "Revenue",
                    "value": 100.0,
                    "unit": "USDm",
                    "period_end": "2023-12-31",
                    "accepted_at": "2023-12-31T00:00:00Z",
                    "as_of_time": "2023-12-31T00:00:00Z",
                    "source_ids": [f"fact-visible-source:{ticker}"],
                },
                {
                    "sec_filing_id": f"fact-future-{ticker}",
                    "entity_id": f"entity-{ticker.lower()}",
                    "ticker": ticker,
                    "concept": "NetIncome",
                    "value": 20.0,
                    "unit": "USDm",
                    "period_end": "2024-03-31",
                    "accepted_at": "2024-01-03T00:00:00Z",
                    "as_of_time": "2024-01-03T00:00:00Z",
                    "source_ids": [f"fact-future-source:{ticker}"],
                },
            )
        ],
    )
    store.write_jsonl(
        "stage=02_normalized/family=market_data/source=yahoo.jsonl",
        [
            market_data
            for ticker in tickers
            for market_data in (
                _market_data(ticker, "2024-01-01", 100.0),
                _market_data(ticker, "2024-01-02", 110.0),
                _market_data(ticker, "2024-01-03", 120.0),
                _market_data(ticker, "2024-01-04", 130.0),
            )
        ],
    )
    store.write_jsonl(
        "stage=02_normalized/family=news/source=fixture.jsonl",
        [
            event
            for ticker in tickers
            for event in (
                {
                    "event_id": f"news-visible-{ticker}",
                    "ticker": ticker,
                    "timestamp": "2024-01-02T00:00:00Z",
                    "headline": f"{ticker} demand improves",
                    "embedding": [0.1, 0.2],
                    "sentiment": 0.7,
                    "importance_score": 0.8,
                    "source_ids": [f"news-visible-source:{ticker}"],
                },
                {
                    "event_id": f"news-future-{ticker}",
                    "ticker": ticker,
                    "timestamp": "2024-01-03T00:00:00Z",
                    "headline": f"{ticker} future event",
                    "embedding": [0.9],
                    "sentiment": 0.1,
                    "importance_score": 0.2,
                    "source_ids": [f"news-future-source:{ticker}"],
                },
            )
        ],
    )
    store.write_jsonl(
        "stage=02_normalized/family=earnings/source=fixture.jsonl",
        [
            event
            for ticker in tickers
            for event in (
                {
                    "earnings_id": f"earnings-visible-{ticker}",
                    "ticker": ticker,
                    "as_of_time": "2024-01-02T00:00:00Z",
                    "surprise": 0.12,
                    "eps_actual": 1.5,
                    "source_ids": [f"earnings-visible-source:{ticker}"],
                },
                {
                    "earnings_id": f"earnings-future-{ticker}",
                    "ticker": ticker,
                    "as_of_time": "2024-01-03T00:00:00Z",
                    "surprise": -0.05,
                    "eps_actual": 1.1,
                    "source_ids": [f"earnings-future-source:{ticker}"],
                },
            )
        ],
    )
    store.write_jsonl(
        "stage=02_normalized/family=macro/source=fixture.jsonl",
        [
            {
                "macro_id": "macro-visible",
                "timestamp": "2024-01-02T00:00:00Z",
                "vix": 18.0,
                "interest_rate": 0.04,
                "source_ids": ["macro-visible-source"],
            },
            {
                "macro_id": "macro-future",
                "timestamp": "2024-01-03T00:00:00Z",
                "vix": 30.0,
                "interest_rate": 0.05,
                "source_ids": ["macro-future-source"],
            },
        ],
    )
    return store


def _market_data(ticker: str, date: str, adjusted_close: float) -> dict[str, object]:
    return {
        "market_data_id": f"market_data-{ticker}-{date}",
        "ticker": ticker,
        "date": date,
        "adjusted_close": adjusted_close,
        "provider": "fixture",
        "source_ids": [f"market_data-source:{ticker}:{date}"],
    }


if __name__ == "__main__":
    unittest.main()
