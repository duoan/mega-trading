import tempfile
import unittest
from pathlib import Path

from mega_trading.core.store import LocalObjectStore
from mega_trading.eval.backtest import BacktestConfig, run_backtest


class BacktestTests(unittest.TestCase):
    def test_backtest_writes_report_with_cost_adjusted_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            labeled_path = "labels/demo/labeled-predictions.jsonl"
            store.write_jsonl(
                labeled_path,
                [
                    _prediction("one", "outperform", "outperform", "low", "low", 0.03),
                    _prediction("two", "underperform", "underperform", "high", "medium", -0.02),
                    _prediction("three", "neutral", "outperform", "low", "low", 0.01),
                ],
            )

            result = run_backtest(
                store,
                backtest_id="bt-demo",
                labeled_prediction_path=labeled_path,
                config=BacktestConfig(fee_bps=1.0, slippage_bps=1.0, confidence_threshold=0.5),
            )
            report = store.read_json(result.report_path)
            manifest = store.read_manifest(result.manifest_path)

            self.assertEqual(report["metrics"]["prediction_count"], 3.0)
            self.assertEqual(report["metrics"]["actionable_signals"], 2.0)
            self.assertEqual(report["metrics"]["return_accuracy"], 2 / 3)
            self.assertEqual(report["metrics"]["risk_accuracy"], 2 / 3)
            self.assertGreater(report["metrics"]["mean_slippage_adjusted_return"], 0.0)
            self.assertIn("sharpe", report["metrics"])
            self.assertEqual(manifest.metadata["prediction_count"], "3")


def _prediction(
    suffix: str,
    pred_return: str,
    actual_return: str,
    pred_risk: str,
    actual_risk: str,
    forward_return: float,
) -> dict[str, object]:
    return {
        "prediction_id": f"pred-{suffix}",
        "prediction_time": "2024-01-02T00:00:00Z",
        "ticker": "ACME",
        "sample_id": f"sample-{suffix}",
        "model_version_id": "model-v1",
        "base_model_version": "base-v1",
        "adapter_version": "adapter-none",
        "head_version": "head-v1",
        "feature_version": "features-v1",
        "label_version": "labels-v1",
        "pred_return_bucket": pred_return,
        "pred_risk_bucket": pred_risk,
        "confidence": 0.8,
        "return_probabilities": {pred_return: 0.8},
        "risk_probabilities": {pred_risk: 0.8},
        "source_ids": [suffix],
        "label_status": "ready",
        "actual_return_bucket": actual_return,
        "actual_risk_bucket": actual_risk,
        "actual_forward_return": forward_return,
        "label_ready_time": "2024-01-22T00:00:00Z",
    }


if __name__ == "__main__":
    unittest.main()
