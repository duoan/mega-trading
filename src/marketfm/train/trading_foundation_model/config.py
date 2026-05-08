"""Configuration and label contracts for TradingFoundationModel."""

from __future__ import annotations

from dataclasses import dataclass

from marketfm.core.hashing import stable_hash


RETURN_LABELS = ("underperform", "neutral", "outperform")
RISK_LABELS = ("low", "medium", "high")
RETURN_TO_ID = {label: index for index, label in enumerate(RETURN_LABELS)}
RISK_TO_ID = {label: index for index, label in enumerate(RISK_LABELS)}


@dataclass(frozen=True)
class TradingFoundationTrainConfig:
    run_id: str
    max_steps: int
    hidden_dim: int = 32
    batch_size: int = 8
    learning_rate: float = 1e-3
    price_window_size: int | None = None
    fundamental_size: int | None = None
    evidence_size: int | None = None
    seed: int = 7
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

    def content_hash(self) -> str:
        return stable_hash(
            {
                "run_id": self.run_id,
                "max_steps": self.max_steps,
                "hidden_dim": self.hidden_dim,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "price_window_size": self.price_window_size,
                "fundamental_size": self.fundamental_size,
                "evidence_size": self.evidence_size,
                "seed": self.seed,
                "device": self.device,
            }
        )


@dataclass(frozen=True)
class TradingFoundationTrainResult:
    steps: int
    checkpoint_path: str
    manifest_path: str
