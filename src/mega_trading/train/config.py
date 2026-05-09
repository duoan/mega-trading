"""Training configuration and label contracts."""

from __future__ import annotations

from dataclasses import dataclass

from mega_trading.core.hashing import stable_hash


RETURN_LABELS = ("underperform", "neutral", "outperform")
RISK_LABELS = ("low", "medium", "high")
RETURN_TO_ID = {label: index for index, label in enumerate(RETURN_LABELS)}
RISK_TO_ID = {label: index for index, label in enumerate(RISK_LABELS)}


@dataclass(frozen=True)
class TradingFoundationTrainConfig:
    run_id: str
    max_steps: int
    hidden_dim: int = 32
    attention_heads: int = 4
    batch_size: int = 8
    learning_rate: float = 1e-3
    validation_fraction: float = 0.2
    eval_interval: int = 10
    market_window_size: int | None = None
    news_size: int | None = None
    sec_filing_size: int | None = None
    earnings_size: int | None = None
    macro_size: int | None = None
    use_market_data: bool = True
    use_news: bool = True
    use_sec_filings: bool = True
    use_earnings: bool = True
    use_macro: bool = True
    seed: int = 7
    device: str = "auto"
    precision: str = "auto"

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.attention_heads <= 0:
            raise ValueError("attention_heads must be positive")
        if self.hidden_dim % self.attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0.0, 1.0)")
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive")
        if not (self.use_market_data or self.use_news or self.use_sec_filings or self.use_earnings or self.use_macro):
            raise ValueError("at least one modality must be enabled")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: auto, cpu, cuda, mps")
        if self.precision not in {"auto", "fp32", "mixed"}:
            raise ValueError("precision must be one of: auto, fp32, mixed")

    def content_hash(self) -> str:
        return stable_hash(
            {
                "run_id": self.run_id,
                "max_steps": self.max_steps,
                "hidden_dim": self.hidden_dim,
                "attention_heads": self.attention_heads,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "validation_fraction": self.validation_fraction,
                "eval_interval": self.eval_interval,
                "market_window_size": self.market_window_size,
                "news_size": self.news_size,
                "sec_filing_size": self.sec_filing_size,
                "earnings_size": self.earnings_size,
                "macro_size": self.macro_size,
                "use_market_data": self.use_market_data,
                "use_news": self.use_news,
                "use_sec_filings": self.use_sec_filings,
                "use_earnings": self.use_earnings,
                "use_macro": self.use_macro,
                "seed": self.seed,
                "device": self.device,
                "precision": self.precision,
            }
        )


@dataclass(frozen=True)
class TradingFoundationTrainResult:
    steps: int
    checkpoint_path: str
    manifest_path: str
