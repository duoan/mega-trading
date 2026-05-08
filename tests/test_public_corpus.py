import tempfile
import unittest
from pathlib import Path

from mega_trading.core.store import LocalObjectStore
from mega_trading.data.corpus import PublicCorpusBuilder, ReadinessError
from mega_trading.data.tokenize import ShardBuilder
from mega_trading.train.cpt import CPTTrainer, TrainConfig
from mega_trading.train.sft import SFTExampleFormatter, SFTTrainer, SFTTrainConfig


class PublicCorpusTests(unittest.TestCase):
    def test_public_corpus_requires_passing_readiness_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            store.write_json("reports/data-readiness.json", {"training_ready": False})

            with self.assertRaises(ReadinessError):
                PublicCorpusBuilder(store).build(mixture_name="public")

    def test_public_corpus_builds_cpt_records_from_enriched_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = _prepared_public_store(tmp)

            result = PublicCorpusBuilder(store).build(mixture_name="public")
            cpt = store.read_jsonl("stage=04_corpus/mixture=public/cpt.jsonl")

            self.assertEqual(result.counts["cpt"], 1)
            self.assertIn("Apple Inc.", cpt[0]["text"])
            self.assertIn("Revenue", cpt[0]["text"])
            self.assertEqual(cpt[0]["as_of_time"], "2024-02-01T00:00:00Z")

    def test_public_corpus_builds_evidence_grounded_sft_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = _prepared_public_store(tmp)

            result = PublicCorpusBuilder(store).build(mixture_name="public")
            sft = store.read_jsonl("stage=04_corpus/mixture=public/sft.jsonl")
            examples = SFTExampleFormatter(store).load("stage=04_corpus/mixture=public/sft.jsonl")

            self.assertEqual(result.counts["sft"], 1)
            self.assertIn("Question:", sft[0]["text"])
            self.assertIn("Answer:", sft[0]["text"])
            self.assertIn("fundamental-f1", sft[0]["evidence_ids"])
            self.assertIn("price-p2", sft[0]["evidence_ids"])
            self.assertIn("investment_view", examples[0].target)

    def test_public_sft_corpus_can_train_sft_smoke_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = _prepared_public_store(tmp)
            PublicCorpusBuilder(store).build(mixture_name="public")

            result = SFTTrainer(store, SFTTrainConfig(run_id="public-sft")).train("stage=04_corpus/mixture=public/sft.jsonl")

            self.assertEqual(result.examples, 1)
            self.assertEqual(store.read_jsonl("runs/public-sft/metrics.jsonl")[0]["schema_compliance"], 1.0)

    def test_public_corpus_shards_can_train_cpt_smoke_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = _prepared_public_store(tmp)
            PublicCorpusBuilder(store).build(mixture_name="public")
            shard = ShardBuilder(store, sequence_length=8).build("public", "cpt")

            result = CPTTrainer(store, TrainConfig(run_id="public-cpt", max_steps=2)).train(shard.shard_path)

            self.assertEqual(result.steps, 2)
            self.assertTrue(store.read_jsonl("runs/public-cpt/metrics.jsonl"))


def _prepared_public_store(tmp: str) -> LocalObjectStore:
    store = LocalObjectStore(Path(tmp))
    store.write_json("reports/data-readiness.json", {"training_ready": True, "quality_score": 1.0})
    store.write_jsonl(
        "stage=03_enriched/company_snapshots.jsonl",
        [
            {
                "snapshot_id": "snapshot-AAPL",
                "entity_id": "sec-1",
                "ticker": "AAPL",
                "company_name": "Apple Inc.",
                "latest_fundamental_as_of_time": "2024-02-01T00:00:00Z",
                "price_start": "2024-01-02",
                "price_end": "2024-01-31",
                "price_observations": 20,
                "source_ids": ["sec:company_tickers:1"],
            }
        ],
    )
    store.write_jsonl(
        "stage=02_normalized/family=fundamentals/source=sec.jsonl",
        [
            {
                "fundamental_id": "f1",
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
        "stage=02_normalized/family=prices/source=yahoo.jsonl",
        [
            {
                "price_id": "p1",
                "ticker": "AAPL",
                "date": "2024-01-02",
                "adjusted_close": 185.0,
                "provider": "yahoo",
                "source_ids": ["p1"],
            },
            {
                "price_id": "p2",
                "ticker": "AAPL",
                "date": "2024-01-31",
                "adjusted_close": 186.0,
                "provider": "yahoo",
                "source_ids": ["p2"],
            },
        ],
    )
    return store


if __name__ == "__main__":
    unittest.main()
