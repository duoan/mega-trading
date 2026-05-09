import unittest

from mega_trading.core.schemas import (
    BacktestResultRecord,
    DelayedLabelRecord,
    ModelVersionRecord,
    PredictionRecord,
    SchemaValidationError,
)
from mega_trading.core.store import ArtifactPaths


class PlatformContractTests(unittest.TestCase):
    def test_model_version_record_is_hashable_and_validated(self) -> None:
        record = _model_version_record()

        self.assertEqual(record.model_version_id, "model-v1")
        self.assertEqual(record.to_dict()["checkpoint_path"], "runs/tfm/checkpoint.pt")
        self.assertEqual(record.content_hash(), _model_version_record().content_hash())

    def test_prediction_record_requires_confidence_range(self) -> None:
        with self.assertRaises(SchemaValidationError):
            PredictionRecord(
                prediction_id="pred-1",
                prediction_time="2024-01-02T00:00:00Z",
                ticker="ACME",
                sample_id="sample-1",
                model_version_id="model-v1",
                base_model_version="base-v1",
                adapter_version="adapter-none",
                head_version="head-v1",
                feature_version="features-v1",
                label_version="labels-v1",
                pred_return_bucket="outperform",
                pred_risk_bucket="low",
                confidence=1.5,
                return_probabilities={"outperform": 1.0},
                risk_probabilities={"low": 1.0},
                source_ids=["sample-1"],
            )

    def test_delayed_label_record_preserves_maturity_time(self) -> None:
        record = DelayedLabelRecord(
            label_id="label-1",
            prediction_id="pred-1",
            ticker="ACME",
            prediction_time="2024-01-02T00:00:00Z",
            label_ready_time="2024-01-22T00:00:00Z",
            actual_return_bucket="neutral",
            actual_risk_bucket="medium",
            actual_forward_return=0.01,
            source_ids=["pred-1"],
        )

        self.assertEqual(record.label_ready_time, "2024-01-22T00:00:00Z")

    def test_backtest_result_requires_metrics(self) -> None:
        with self.assertRaises(SchemaValidationError):
            BacktestResultRecord(
                backtest_id="bt-1",
                created_at="2024-01-22T00:00:00Z",
                prediction_path="predictions/replay/predictions.jsonl",
                labeled_prediction_path="labels/replay/labeled-predictions.jsonl",
                metrics={},
                config={"fee_bps": 1.0},
                source_ids=["labels/replay/labeled-predictions.jsonl"],
            )

    def test_platform_artifact_paths_are_stable(self) -> None:
        paths = ArtifactPaths(run_id="demo")

        self.assertEqual(paths.model("model-v1"), "models/model-v1/model-version.json")
        self.assertEqual(paths.predictions("replay-1"), "predictions/replay-1/predictions.jsonl")
        self.assertEqual(paths.labels("labels-1"), "labels/labels-1/labels.jsonl")
        self.assertEqual(paths.eval("bt-1", "backtest-report"), "evals/bt-1/backtest-report.json")
        self.assertEqual(paths.replay("replay-1", "events"), "replay/replay-1/events.jsonl")


def _model_version_record() -> ModelVersionRecord:
    return ModelVersionRecord(
        model_version_id="model-v1",
        run_id="tfm",
        checkpoint_path="runs/tfm/checkpoint.pt",
        manifest_path="manifests/runs/tfm-trading-foundation-model.json",
        metrics_path="runs/tfm/metrics.jsonl",
        base_model_version="base-v1",
        adapter_version="adapter-none",
        head_version="head-v1",
        feature_version="features-v1",
        label_version="labels-v1",
        data_snapshot_version="public-v1",
        config_hash="abc123",
        created_at="2024-01-02T00:00:00Z",
    )


if __name__ == "__main__":
    unittest.main()
