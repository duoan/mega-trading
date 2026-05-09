import tempfile
import unittest
from pathlib import Path

from mega_trading.core.store import LocalObjectStore
from mega_trading.data.shards import MalformedSampleError, StreamShardBuilder


class ShardBuilderTests(unittest.TestCase):
    def test_stream_shard_builder_writes_compact_model_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            store.write_jsonl("stage=04_corpus/mixture=public/partitions/ticker=ACME/samples.jsonl", [_sample()])
            store.write_jsonl(
                "stage=04_corpus/mixture=public/partitions/ticker=BETA/samples.jsonl",
                [_sample("sample-BETA-2024-01-03", ticker="BETA")],
            )

            result = StreamShardBuilder(store, num_workers=2).build("public")
            rows = store.read_jsonl(result.shard_path)
            manifest = store.read_manifest(result.manifest_path)

            self.assertEqual(result.num_samples, 2)
            self.assertEqual(rows[0]["sample_id"], "sample-ACME-2024-01-02")
            self.assertEqual(rows[0]["return_label"], "outperform")
            self.assertEqual(rows[0]["risk_label"], "low")
            self.assertEqual(rows[0]["market_returns"], [0.0, 0.1])
            self.assertEqual(rows[0]["sec_filing_features"], [100.0])
            self.assertEqual(rows[0]["earnings_features"], [0.12, 1.5])
            self.assertEqual(rows[0]["news_embeddings"], [0.3, 0.8])
            self.assertEqual(rows[0]["macro_features"], [18.0, 0.04])
            self.assertEqual(manifest.metadata["artifact"], "multi_stream_samples")
            self.assertEqual(manifest.metadata["workers"], "2")

    def test_malformed_sample_fails_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            store.write_jsonl("stage=04_corpus/mixture=public/samples.jsonl", [{"sample_id": "bad"}])

            with self.assertRaises(MalformedSampleError):
                StreamShardBuilder(store).build("public")


def _sample(sample_id: str = "sample-ACME-2024-01-02", ticker: str = "ACME") -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "ticker": ticker,
        "as_of_time": "2024-01-02T00:00:00Z",
        "market_data_window": [
            {"date": "2024-01-01", "adjusted_close": 100.0, "market_data_id": "px-1", "source_ids": ["px-1"]},
            {"date": "2024-01-02", "adjusted_close": 110.0, "market_data_id": "px-2", "source_ids": ["px-2"]},
        ],
        "news_window": [{"event_id": "news-1", "embedding": [0.3], "importance_score": 0.8, "source_ids": ["news-1"]}],
        "sec_filing_window": [
            {
                "sec_filing_id": "fact-1",
                "concept": "Revenue",
                "value": 100.0,
                "as_of_time": "2023-12-31T00:00:00Z",
                "source_ids": ["fact-1"],
            }
        ],
        "earnings_window": [
            {
                "earnings_id": "earnings-1",
                "as_of_time": "2024-01-01T00:00:00Z",
                "surprise": 0.12,
                "eps_actual": 1.5,
                "source_ids": ["earnings-1"],
            }
        ],
        "macro_window": [{"timestamp": "2024-01-02T00:00:00Z", "vix": 18.0, "interest_rate": 0.04}],
        "labels": {"forward_return_bucket": "outperform", "risk_bucket": "low", "forward_return": 0.08},
        "source_ids": ["px-1", "px-2", "fact-1", "news-1"],
    }


if __name__ == "__main__":
    unittest.main()
