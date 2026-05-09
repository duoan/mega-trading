"""Local model-server wrapper around registered TradingFoundationModel checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from mega_trading.core.store import LocalObjectStore
from mega_trading.train.inference import ModelPredictor


@dataclass(frozen=True)
class FeatureRequest:
    sample_id: str
    ticker: str
    as_of_time: str
    features: dict[str, object]
    source_ids: list[str]

    @classmethod
    def from_shard_row(cls, row: dict[str, object]) -> "FeatureRequest":
        return cls(
            sample_id=str(row["sample_id"]),
            ticker=str(row["ticker"]),
            as_of_time=str(row["as_of_time"]),
            features=row,
            source_ids=[str(source_id) for source_id in row.get("source_ids", [])] or [str(row["sample_id"])],
        )


@dataclass(frozen=True)
class PredictionResponse:
    prediction_id: str
    prediction_time: str
    ticker: str
    model_version_id: str
    pred_return_bucket: str
    pred_risk_bucket: str
    confidence: float
    return_probabilities: dict[str, float]
    risk_probabilities: dict[str, float]
    source_ids: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ModelServer:
    def __init__(self, store: LocalObjectStore, model_version_id: str, device: str = "auto") -> None:
        self.predictor = ModelPredictor(store, model_version_id=model_version_id, device=device)

    def predict(self, request: FeatureRequest) -> PredictionResponse:
        prediction = self.predictor.predict_row(
            {
                **request.features,
                "sample_id": request.sample_id,
                "ticker": request.ticker,
                "as_of_time": request.as_of_time,
                "source_ids": request.source_ids,
            }
        )
        model_version = self.predictor.model_version
        return PredictionResponse(
            prediction_id=f"serve-{request.sample_id}",
            prediction_time=request.as_of_time,
            ticker=request.ticker,
            model_version_id=model_version.model_version_id,
            pred_return_bucket=prediction.return_bucket,
            pred_risk_bucket=prediction.risk_bucket,
            confidence=prediction.confidence,
            return_probabilities=prediction.return_probabilities,
            risk_probabilities=prediction.risk_probabilities,
            source_ids=request.source_ids,
        )
