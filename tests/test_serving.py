import tempfile
import unittest
from pathlib import Path

from mega_trading.cli import main
from mega_trading.core.store import LocalObjectStore
from mega_trading.serving import FeatureRequest, ModelServer
from mega_trading.train.config import TradingFoundationTrainConfig
from mega_trading.train.trainer import TradingFoundationTrainer


class ServingTests(unittest.TestCase):
    def test_model_server_predicts_from_feature_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store, shard_path, model_version_id = _trained_store(tmp)
            row = store.read_jsonl(shard_path)[0]

            response = ModelServer(store, model_version_id, device="cpu").predict(FeatureRequest.from_shard_row(row))

            self.assertEqual(response.ticker, "ACME")
            self.assertEqual(response.model_version_id, model_version_id)
            self.assertIn(response.pred_return_bucket, {"underperform", "neutral", "outperform"})

    def test_serve_smoke_command_writes_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store, shard_path, model_version_id = _trained_store(tmp)

            exit_code = main(
                [
                    "serve-smoke",
                    f"--data-dir={tmp}",
                    f"--model-version-id={model_version_id}",
                    f"--shard-path={shard_path}",
                    "--device=cpu",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((Path(tmp) / "replay/serve-smoke/response.json").exists())
            response = store.read_json("replay/serve-smoke/response.json")
            self.assertEqual(response["model_version_id"], model_version_id)


def _trained_store(tmp: str) -> tuple[LocalObjectStore, str, str]:
    store = LocalObjectStore(Path(tmp))
    shard_path = "stage=05_shards/mixture=public/samples.jsonl"
    store.write_jsonl(
        shard_path,
        [
            _row("a", "outperform", "low"),
            _row("b", "underperform", "high"),
        ],
    )
    result = TradingFoundationTrainer(
        store,
        TradingFoundationTrainConfig(
            run_id="tfm-serving",
            max_steps=1,
            hidden_dim=8,
            batch_size=2,
            validation_fraction=0.0,
            device="cpu",
        ),
    ).train(shard_path)
    manifest = store.read_manifest(result.manifest_path)
    return store, shard_path, str(manifest.metadata["model_version_id"])


def _row(sample_id: str, return_label: str, risk_label: str) -> dict[str, object]:
    return {
        "sample_id": f"sample-{sample_id}",
        "ticker": "ACME",
        "as_of_time": "2024-01-02T00:00:00Z",
        "price_returns": [0.0, 0.01, -0.02],
        "price_levels": [0.0, 0.01, -0.01],
        "fundamental_values": [100.0],
        "evidence_token_ids": [2, 3, 4],
        "return_label": return_label,
        "risk_label": risk_label,
        "source_ids": [sample_id],
        "evidence_ids": [],
    }


if __name__ == "__main__":
    unittest.main()
