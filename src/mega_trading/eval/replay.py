"""Historical replay inference over stream shards."""

from __future__ import annotations

from dataclasses import dataclass

from mega_trading.core.schemas import PredictionRecord
from mega_trading.core.store import LocalObjectStore
from mega_trading.eval.prediction_log import write_prediction_log
from mega_trading.train.inference import ModelPredictor


@dataclass(frozen=True)
class ReplayResult:
    prediction_path: str
    manifest_path: str
    predictions: int


def run_replay_inference(
    store: LocalObjectStore,
    *,
    replay_id: str,
    model_version_id: str,
    shard_path: str,
    max_predictions: int | None = None,
    device: str = "auto",
) -> ReplayResult:
    predictor = ModelPredictor(store, model_version_id=model_version_id, device=device)
    result = write_prediction_log(
        store,
        replay_id=replay_id,
        records=_prediction_records(store, predictor, replay_id, shard_path, max_predictions),
        metadata={
            "model_version_id": model_version_id,
            "shard_path": shard_path,
        },
    )
    return ReplayResult(
        prediction_path=result.prediction_path,
        manifest_path=result.manifest_path,
        predictions=result.predictions,
    )


def _prediction_records(
    store: LocalObjectStore,
    predictor: ModelPredictor,
    replay_id: str,
    shard_path: str,
    max_predictions: int | None,
):
    for index, row in enumerate(store.iter_jsonl(shard_path), start=1):
        if max_predictions is not None and index > max_predictions:
            break
        prediction = predictor.predict_row(row)
        model_version = predictor.model_version
        prediction_id = f"{replay_id}-{row['sample_id']}"
        record = PredictionRecord(
            prediction_id=prediction_id,
            prediction_time=str(row["as_of_time"]),
            ticker=str(row["ticker"]),
            sample_id=str(row["sample_id"]),
            model_version_id=model_version.model_version_id,
            base_model_version=model_version.base_model_version,
            adapter_version=model_version.adapter_version,
            head_version=model_version.head_version,
            feature_version=model_version.feature_version,
            label_version=model_version.label_version,
            pred_return_bucket=prediction.return_bucket,
            pred_risk_bucket=prediction.risk_bucket,
            confidence=prediction.confidence,
            return_probabilities=prediction.return_probabilities,
            risk_probabilities=prediction.risk_probabilities,
            source_ids=[str(source_id) for source_id in row.get("source_ids", [])] or [str(row["sample_id"])],
            metadata={
                "replay_id": replay_id,
                "row_index": index,
            },
        )
        yield record
