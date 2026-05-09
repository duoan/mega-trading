"""Delayed label materialization for replay prediction logs."""

from __future__ import annotations

from dataclasses import dataclass

from mega_trading.core.schemas import DelayedLabelRecord, Manifest
from mega_trading.core.store import ArtifactPaths, LocalObjectStore


@dataclass(frozen=True)
class DelayedLabelResult:
    label_path: str
    labeled_prediction_path: str
    manifest_path: str
    labels: int


def materialize_delayed_labels(
    store: LocalObjectStore,
    *,
    label_run_id: str,
    prediction_path: str,
    shard_path: str,
) -> DelayedLabelResult:
    labels_by_sample = {str(row["sample_id"]): row for row in store.iter_jsonl(shard_path)}
    labels: list[dict[str, object]] = []
    labeled_predictions: list[dict[str, object]] = []
    for prediction in store.iter_jsonl(prediction_path):
        sample_id = str(prediction["sample_id"])
        sample = labels_by_sample.get(sample_id)
        if sample is None:
            labeled_predictions.append({**prediction, "label_status": "missing"})
            continue
        label_ready_time = str(sample.get("label_ready_time") or sample.get("as_of_time") or prediction["prediction_time"])
        record = DelayedLabelRecord(
            label_id=f"{label_run_id}-{prediction['prediction_id']}",
            prediction_id=str(prediction["prediction_id"]),
            ticker=str(prediction["ticker"]),
            prediction_time=str(prediction["prediction_time"]),
            label_ready_time=label_ready_time,
            actual_return_bucket=str(sample["return_label"]),
            actual_risk_bucket=str(sample["risk_label"]),
            actual_forward_return=float(sample.get("forward_return", 0.0)),
            source_ids=[str(source_id) for source_id in sample.get("source_ids", [])] or [sample_id],
            metadata={"sample_id": sample_id},
        )
        labels.append(record.to_dict())
        labeled_predictions.append(
            {
                **prediction,
                "label_status": "ready",
                "actual_return_bucket": record.actual_return_bucket,
                "actual_risk_bucket": record.actual_risk_bucket,
                "actual_forward_return": record.actual_forward_return,
                "label_ready_time": record.label_ready_time,
            }
        )

    paths = ArtifactPaths()
    label_path = paths.labels(label_run_id)
    labeled_prediction_path = paths.labels(label_run_id, "labeled-predictions")
    store.write_jsonl(label_path, labels)
    store.write_jsonl(labeled_prediction_path, labeled_predictions)
    manifest = Manifest(
        manifest_id=f"{label_run_id}-delayed-labels",
        artifact_type="delayed_labels",
        paths=[label_path, labeled_prediction_path],
        metadata={
            "label_run_id": label_run_id,
            "prediction_path": prediction_path,
            "shard_path": shard_path,
            "labels": str(len(labels)),
        },
    )
    manifest_path = paths.manifest("labels", f"{label_run_id}-delayed-labels")
    store.write_manifest(manifest_path, manifest)
    return DelayedLabelResult(
        label_path=label_path,
        labeled_prediction_path=labeled_prediction_path,
        manifest_path=manifest_path,
        labels=len(labels),
    )
