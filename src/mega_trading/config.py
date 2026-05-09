"""Configuration contracts for Mega-Trading."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BuildConfig:
    mixture_name: str = "public"
    source: str = "yahoo"
    block_size: int = 128
    stride: int = 64
    min_events_per_ticker: int = 32
    max_tickers: int | None = None

    def __post_init__(self) -> None:
        if self.block_size < 8:
            raise ValueError("block_size must be at least 8")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.min_events_per_ticker < 2:
            raise ValueError("min_events_per_ticker must be at least 2")
        if self.max_tickers is not None and self.max_tickers <= 0:
            raise ValueError("max_tickers must be positive when set")


@dataclass(frozen=True)
class TrainConfig:
    run_id: str
    mixture_name: str = "public"
    max_steps: int = 100
    batch_size: int = 32
    learning_rate: float = 3e-4
    validation_fraction: float = 0.1
    eval_interval: int = 20
    hidden_dim: int = 128
    layers: int = 4
    attention_heads: int = 4
    dropout: float = 0.1
    seed: int = 7
    device: str = "auto"
    precision: str = "auto"
    wandb_enabled: bool = True
    wandb_project: str = "mega-trading"
    wandb_entity: str | None = None
    wandb_mode: str = "offline"
    progress_bar: bool = True

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0.0, 1.0)")
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.layers <= 0:
            raise ValueError("layers must be positive")
        if self.attention_heads <= 0:
            raise ValueError("attention_heads must be positive")
        if self.hidden_dim % self.attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0)")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: auto, cpu, cuda, mps")
        if self.precision not in {"auto", "fp32", "mixed"}:
            raise ValueError("precision must be one of: auto, fp32, mixed")
        if self.wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb_mode must be one of: online, offline, disabled")
