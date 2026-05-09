"""Append-only prediction log artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from mega_trading.core.schemas import Manifest, PredictionRecord
from mega_trading.core.store import ArtifactPaths, LocalObjectStore


@dataclass(frozen=True)
class PredictionLogResult:
    prediction_path: str
    manifest_path: str
    predictions: int


def write_prediction_log(
    store: LocalObjectStore,
    *,
    replay_id: str,
    records: Iterable[PredictionRecord],
    metadata: dict[str, str] | None = None,
) -> PredictionLogResult:
    paths = ArtifactPaths()
    prediction_path = paths.predictions(replay_id)
    count = store.write_jsonl_iter(prediction_path, (record.to_dict() for record in records))
    manifest = Manifest(
        manifest_id=f"{replay_id}-prediction-log",
        artifact_type="prediction_log",
        paths=[prediction_path],
        metadata={
            "replay_id": replay_id,
            "predictions": str(count),
            **(metadata or {}),
        },
    )
    manifest_path = paths.manifest("predictions", f"{replay_id}-prediction-log")
    store.write_manifest(manifest_path, manifest)
    return PredictionLogResult(prediction_path=prediction_path, manifest_path=manifest_path, predictions=count)
