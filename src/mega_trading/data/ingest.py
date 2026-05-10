"""Source request contracts for training data preparation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BinanceTradesIngestRequest:
    symbols: tuple[str, ...]
    start: str
    end: str
    frequency: str = "monthly"
    download_workers: int = 4
