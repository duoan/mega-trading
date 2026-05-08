import tempfile
import unittest
from pathlib import Path

from marketfm.core.store import LocalObjectStore
from marketfm.data.labels import LabelConfig
from marketfm.data.samples import MultiStreamSampleBuilder


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
            self.assertEqual([row["fundamental_id"] for row in sample["fundamental_facts"]], ["fact-visible"])
            self.assertNotIn("fact-future", sample["source_ids"])
            self.assertIn("stage=04_corpus/mixture=public/samples.jsonl", manifest.paths)

    def test_requires_readiness_report_to_be_training_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            store.write_json("reports/data-readiness.json", {"training_ready": False})

            with self.assertRaises(ValueError):
                MultiStreamSampleBuilder(store).build(mixture_name="public")


def _prepared_store(tmp: str) -> LocalObjectStore:
    store = LocalObjectStore(Path(tmp))
    store.write_json("reports/data-readiness.json", {"training_ready": True, "quality_score": 1.0})
    store.write_jsonl(
        "stage=02_normalized/family=entities/source=sec.jsonl",
        [{"entity_id": "entity-acme", "ticker": "ACME", "company_name": "Acme Inc.", "source_ids": ["entity-source"]}],
    )
    store.write_jsonl(
        "stage=02_normalized/family=fundamentals/source=sec.jsonl",
        [
            {
                "fundamental_id": "fact-visible",
                "entity_id": "entity-acme",
                "ticker": "ACME",
                "concept": "Revenue",
                "value": 100.0,
                "unit": "USDm",
                "period_end": "2023-12-31",
                "accepted_at": "2023-12-31T00:00:00Z",
                "as_of_time": "2023-12-31T00:00:00Z",
                "source_ids": ["fact-visible-source"],
            },
            {
                "fundamental_id": "fact-future",
                "entity_id": "entity-acme",
                "ticker": "ACME",
                "concept": "NetIncome",
                "value": 20.0,
                "unit": "USDm",
                "period_end": "2024-03-31",
                "accepted_at": "2024-01-03T00:00:00Z",
                "as_of_time": "2024-01-03T00:00:00Z",
                "source_ids": ["fact-future-source"],
            },
        ],
    )
    store.write_jsonl(
        "stage=02_normalized/family=prices/source=yahoo.jsonl",
        [
            _price("ACME", "2024-01-01", 100.0),
            _price("ACME", "2024-01-02", 110.0),
            _price("ACME", "2024-01-03", 120.0),
            _price("ACME", "2024-01-04", 130.0),
        ],
    )
    store.write_jsonl(
        "stage=02_normalized/family=evidence/source=fixture.jsonl",
        [
            {
                "evidence_id": "ev-visible",
                "entity_id": "entity-acme",
                "ticker": "ACME",
                "source_type": "10-K",
                "document_id": "doc-visible",
                "timestamp": "2023-12-31T00:00:00Z",
                "as_of_time": "2023-12-31T00:00:00Z",
                "text": "ACME reported improving fundamentals.",
                "uri": "fixture://acme",
            }
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
