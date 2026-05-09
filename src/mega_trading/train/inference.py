"""Checkpoint-backed inference for replay and serving paths."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mega_trading.core.registry import ModelRegistry
from mega_trading.core.schemas import ModelVersionRecord
from mega_trading.core.store import LocalObjectStore
from mega_trading.train.config import RETURN_LABELS, RISK_LABELS
from mega_trading.train.dataset import row_to_item
from mega_trading.train.model import TradingFoundationModel
from mega_trading.train.trainer import _resolve_device


@dataclass(frozen=True)
class ModelPrediction:
    return_bucket: str
    risk_bucket: str
    confidence: float
    return_probabilities: dict[str, float]
    risk_probabilities: dict[str, float]


class ModelPredictor:
    def __init__(self, store: LocalObjectStore, model_version_id: str, device: str = "auto") -> None:
        self.store = store
        self.model_version = ModelRegistry(store).read_model_version(model_version_id)
        self.device = _resolve_device(device)
        self.model = _load_model(store, self.model_version, self.device)

    def predict_row(self, row: dict[str, object]) -> ModelPrediction:
        item = row_to_item(_row_with_dummy_labels(row), self.model.price_window_size, self.model.fundamental_size, self.model.evidence_size)
        batch = {
            "price": item["price"].unsqueeze(0).to(self.device),
            "fundamentals": item["fundamentals"].unsqueeze(0).to(self.device),
            "evidence": item["evidence"].unsqueeze(0).to(self.device),
        }
        with torch.no_grad():
            return_logits, risk_logits = self.model(batch)
            return_probs = torch.softmax(return_logits, dim=-1)[0].detach().cpu().tolist()
            risk_probs = torch.softmax(risk_logits, dim=-1)[0].detach().cpu().tolist()
        return_distribution = dict(zip(RETURN_LABELS, return_probs, strict=True))
        risk_distribution = dict(zip(RISK_LABELS, risk_probs, strict=True))
        return_bucket = max(return_distribution, key=return_distribution.get)
        risk_bucket = max(risk_distribution, key=risk_distribution.get)
        confidence = max(return_distribution[return_bucket], risk_distribution[risk_bucket])
        return ModelPrediction(
            return_bucket=return_bucket,
            risk_bucket=risk_bucket,
            confidence=float(confidence),
            return_probabilities={key: float(value) for key, value in return_distribution.items()},
            risk_probabilities={key: float(value) for key, value in risk_distribution.items()},
        )


def _load_model(store: LocalObjectStore, model_version: ModelVersionRecord, device: torch.device) -> TradingFoundationModel:
    checkpoint = torch.load(store.root / model_version.checkpoint_path, map_location="cpu")
    config = dict(checkpoint["config"])
    stream_sizes = dict(checkpoint["stream_sizes"])
    model = TradingFoundationModel(
        price_window_size=int(stream_sizes["price_window_size"]),
        fundamental_size=int(stream_sizes["fundamental_size"]),
        evidence_size=int(stream_sizes["evidence_size"]),
        hidden_dim=int(config["hidden_dim"]),
        attention_heads=int(config["attention_heads"]),
        use_price=bool(config["use_price"]),
        use_fundamentals=bool(config["use_fundamentals"]),
        use_evidence=bool(config["use_evidence"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _row_with_dummy_labels(row: dict[str, object]) -> dict[str, object]:
    return {
        **row,
        "return_label": str(row.get("return_label", "neutral")),
        "risk_label": str(row.get("risk_label", "low")),
    }
