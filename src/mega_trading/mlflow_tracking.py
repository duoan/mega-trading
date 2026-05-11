"""MLflow logging helpers for evaluation artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from mega_trading.core.store import ArtifactNotFoundError, LocalObjectStore


@dataclass(frozen=True)
class MlflowRunConfig:
    enabled: bool
    experiment: str
    tracking_uri: str | None = None


def default_tracking_uri(store: LocalObjectStore) -> str:
    return f"sqlite:///{(store.root / 'runs' / 'mlflow' / 'mlflow.db').resolve()}"


def log_evaluation_artifacts(
    store: LocalObjectStore,
    *,
    run_id: str,
    config: MlflowRunConfig,
    metric_source_path: str | None = None,
    artifact_paths: list[str] | tuple[str, ...] = (),
    tags: dict[str, str] | None = None,
) -> None:
    """Log backtest metrics and local artifacts to the run's MLflow experiment."""
    if not config.enabled:
        return

    tracking_uri = config.tracking_uri or default_tracking_uri(store)
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.experiment)
    run_tags = {
        "project": "mega-trading",
        "run_id": run_id,
        "stage": "evaluation",
        **{str(key): str(value) for key, value in (tags or {}).items()},
    }
    with _evaluation_run(mlflow, config.experiment, run_id, run_tags):
        mlflow.set_tags(run_tags)
        if metric_source_path:
            for key, value in _metric_values(store.read_json(metric_source_path)).items():
                mlflow.log_metric(key, value)
        for artifact_path in artifact_paths:
            target = _artifact_target(store, artifact_path)
            if target.exists():
                mlflow.log_artifact(str(target), artifact_path=_artifact_group(artifact_path))


def _evaluation_run(mlflow: Any, experiment: str, run_id: str, tags: dict[str, str]):
    existing_run_id = _find_existing_run_id(mlflow, experiment, run_id)
    if existing_run_id:
        return mlflow.start_run(run_id=existing_run_id)
    return mlflow.start_run(run_name=f"{run_id}-evaluation", tags=tags)


def _find_existing_run_id(mlflow: Any, experiment: str, run_id: str) -> str | None:
    try:
        matches = mlflow.search_runs(
            experiment_names=[experiment],
            filter_string=f"tags.mlflow.runName = '{run_id}'",
            output_format="list",
            max_results=1,
        )
    except Exception:
        return None
    if not matches:
        return None
    info = getattr(matches[0], "info", None)
    return str(getattr(info, "run_id", "")) or None


def _metric_values(report: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in report.items():
        metric_name = _metric_name(key)
        if metric_name is None:
            continue
        number = _as_float(value)
        if number is not None:
            metrics[metric_name] = number
    return metrics


def _metric_name(key: str) -> str | None:
    if key.startswith("backtest_"):
        return "backtest/" + key.removeprefix("backtest_")
    if key == "price_depth_distribution_l1":
        return "backtest/price_depth_distribution_l1"
    return None


def _as_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _artifact_target(store: LocalObjectStore, path: str) -> Path:
    try:
        return store._resolve(path)
    except ValueError:
        raise
    except ArtifactNotFoundError:
        raise


def _artifact_group(path: str) -> str:
    parts = Path(path).parts
    if len(parts) >= 2:
        return "/".join(parts[:-1])
    return "artifacts"
