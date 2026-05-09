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
            self.assertEqual([row["date"] for row in sample["price_window"]], ["2024-01-01", "2024-01-02"])
            self.assertEqual(sample["labels"]["label_start_date"], "2024-01-03")
            self.assertEqual(sample["labels"]["forward_return_bucket"], "outperform")
            self.assertEqual([row["fundamental_id"] for row in sample["fundamental_facts"]], ["fact-visible-ACME"])
            self.assertNotIn("fact-future", sample["source_ids"])
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
            samples = store.read_jsonl("stage=04_corpus/mixture=public/samples.jsonl")
            manifest = store.read_manifest(result.manifest_path)

            self.assertEqual(result.samples, 2)
            self.assertEqual({sample["ticker"] for sample in samples}, {"ACME", "BETA"})
            self.assertEqual(manifest.metadata["workers"], "2")


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
        "stage=02_normalized/family=fundamentals/source=sec.jsonl",
        [
            fact
            for ticker in tickers
            for fact in (
                {
                    "fundamental_id": f"fact-visible-{ticker}",
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
                    "fundamental_id": f"fact-future-{ticker}",
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
        "stage=02_normalized/family=prices/source=yahoo.jsonl",
        [
            price
            for ticker in tickers
            for price in (
                _price(ticker, "2024-01-01", 100.0),
                _price(ticker, "2024-01-02", 110.0),
                _price(ticker, "2024-01-03", 120.0),
                _price(ticker, "2024-01-04", 130.0),
            )
        ],
    )
    store.write_jsonl(
        "stage=02_normalized/family=evidence/source=fixture.jsonl",
        [
            {
                "evidence_id": f"ev-visible-{ticker}",
                "entity_id": f"entity-{ticker.lower()}",
                "ticker": ticker,
                "source_type": "10-K",
                "document_id": f"doc-visible-{ticker}",
                "timestamp": "2023-12-31T00:00:00Z",
                "as_of_time": "2023-12-31T00:00:00Z",
                "text": f"{ticker} reported improving fundamentals.",
                "uri": f"fixture://{ticker.lower()}",
            }
            for ticker in tickers
        ],
    )
    return store


def _price(ticker: str, date: str, adjusted_close: float) -> dict[str, object]:
    return {
        "price_id": f"price-{ticker}-{date}",
        "ticker": ticker,
        "date": date,
        "adjusted_close": adjusted_close,
        "provider": "fixture",
        "source_ids": [f"price-source:{ticker}:{date}"],
    }


if __name__ == "__main__":
    unittest.main()
