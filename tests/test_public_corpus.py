import tempfile
import unittest
from pathlib import Path

from marketfm.core.store import LocalObjectStore
from marketfm.data.corpus import PublicCorpusBuilder, ReadinessError
from marketfm.data.tokenize import ShardBuilder
from marketfm.train.cpt import CPTTrainer, TrainConfig


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
            self.assertEqual(cpt[0]["as_of_time"], "2024-02-01T00:00:00Z")

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
    return store


if __name__ == "__main__":
    unittest.main()
