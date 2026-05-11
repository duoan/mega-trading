import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mega_trading.core.store import LocalObjectStore
from mega_trading.mlflow_tracking import MlflowRunConfig, log_evaluation_artifacts


class FakeMlflowRun:
    def __init__(self, fake, run_id: str) -> None:
        self.fake = fake
        self.info = type("Info", (), {"run_id": run_id})()

    def __enter__(self):
        self.fake.active_run_ids.append(self.info.run_id)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class FakeMlflow:
    def __init__(self) -> None:
        self.tracking_uri = None
        self.experiment = None
        self.metrics: dict[str, float] = {}
        self.artifacts: list[str] = []
        self.tags: dict[str, str] = {}
        self.active_run_ids: list[str] = []

    def set_tracking_uri(self, tracking_uri: str) -> None:
        self.tracking_uri = tracking_uri

    def set_experiment(self, experiment: str) -> None:
        self.experiment = experiment

    def search_runs(self, **_kwargs):
        return []

    def start_run(self, **kwargs):
        self.tags.update({str(key): str(value) for key, value in dict(kwargs.get("tags", {})).items()})
        return FakeMlflowRun(self, "evaluation-run")

    def set_tags(self, tags: dict[str, str]) -> None:
        self.tags.update({str(key): str(value) for key, value in tags.items()})

    def log_metric(self, key: str, value: float) -> None:
        self.metrics[key] = value

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        suffix = f":{artifact_path}" if artifact_path else ""
        self.artifacts.append(f"{local_path}{suffix}")


class MlflowTrackingTests(unittest.TestCase):
    def test_disabled_mlflow_does_not_import_or_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            store.write_json("evals/run-a/backtest.json", {"backtest_loss": 1.0})

            with patch.dict(sys.modules, {"mlflow": None}):
                log_evaluation_artifacts(
                    store,
                    run_id="run-a",
                    config=MlflowRunConfig(enabled=False, experiment="exp", tracking_uri=None),
                    metric_source_path="evals/run-a/backtest.json",
                    artifact_paths=["evals/run-a/backtest.json"],
                )

    def test_logs_backtest_metrics_and_artifacts_to_mlflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalObjectStore(root)
            store.write_json(
                "evals/run-a/backtest.json",
                {
                    "backtest_loss": 0.5,
                    "backtest_perplexity": 1.65,
                    "backtest_top1_accuracy": 0.25,
                    "price_depth_distribution_l1": 0.125,
                    "stage": "backtest",
                },
            )
            report = root / "reports/run-a/backtest.html"
            report.parent.mkdir(parents=True)
            report.write_text("<html></html>", encoding="utf-8")
            fake_mlflow = FakeMlflow()

            with patch.dict(sys.modules, {"mlflow": fake_mlflow}):
                log_evaluation_artifacts(
                    store,
                    run_id="run-a",
                    config=MlflowRunConfig(
                        enabled=True,
                        experiment="mega-test",
                        tracking_uri="sqlite:///tmp/mlflow.db",
                    ),
                    metric_source_path="evals/run-a/backtest.json",
                    artifact_paths=["evals/run-a/backtest.json", "reports/run-a/backtest.html"],
                    tags={"ablation.name": "smoke"},
                )

            self.assertEqual(fake_mlflow.tracking_uri, "sqlite:///tmp/mlflow.db")
            self.assertEqual(fake_mlflow.experiment, "mega-test")
            self.assertEqual(fake_mlflow.metrics["backtest/loss"], 0.5)
            self.assertEqual(fake_mlflow.metrics["backtest/perplexity"], 1.65)
            self.assertEqual(fake_mlflow.metrics["backtest/top1_accuracy"], 0.25)
            self.assertEqual(fake_mlflow.metrics["backtest/price_depth_distribution_l1"], 0.125)
            self.assertEqual(fake_mlflow.tags["run_id"], "run-a")
            self.assertEqual(fake_mlflow.tags["ablation.name"], "smoke")
            self.assertTrue(any("evals/run-a/backtest.json" in path for path in fake_mlflow.artifacts))
            self.assertTrue(any("reports/run-a/backtest.html" in path for path in fake_mlflow.artifacts))


if __name__ == "__main__":
    unittest.main()
