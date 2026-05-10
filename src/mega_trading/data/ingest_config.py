"""Config-driven ingestion contracts for paper-style event data."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PAPER_SOURCES = {"fixture", "hf_ohlcv_1m", "binance_trades"}


@dataclass(frozen=True)
class IngestSourceConfig:
    name: str
    tickers: tuple[str, ...] = ()
    enabled: bool = True
    start: str | None = None
    end: str | None = None
    frequency: str = "monthly"
    download_workers: int = 4

    def __post_init__(self) -> None:
        if self.name not in PAPER_SOURCES:
            raise ValueError(f"unsupported ingest source for paper pipeline: {self.name}")
        if self.name in {"hf_ohlcv_1m", "binance_trades"}:
            if not self.tickers:
                raise ValueError(f"{self.name} requires tickers")
            if not self.start or not self.end:
                raise ValueError(f"{self.name} requires start and end")
        if self.name == "binance_trades" and self.frequency not in {"daily", "monthly"}:
            raise ValueError("binance_trades frequency must be daily or monthly")
        if self.name == "binance_trades" and self.download_workers <= 0:
            raise ValueError("binance_trades download_workers must be positive")


@dataclass(frozen=True)
class IngestPipelineConfig:
    output_dir: str
    sources: tuple[IngestSourceConfig, ...]
    quality_enabled: bool = True
    quality_fail_on_error: bool = False

    def __post_init__(self) -> None:
        if not self.output_dir:
            raise ValueError("ingest output_dir is required")
        if not self.sources:
            raise ValueError("at least one ingest source is required")

    @classmethod
    def from_dict(cls, value: dict[str, Any], base_path: Path | None = None) -> "IngestPipelineConfig":
        config_base_path = base_path or Path.cwd()
        sources = tuple(_source_from_dict(source, config_base_path) for source in value.get("sources", []))
        quality = value.get("quality", {})
        if not isinstance(quality, dict):
            quality = {}
        return cls(
            output_dir=str(value.get("output_dir", "")),
            sources=sources,
            quality_enabled=bool(quality.get("enabled", True)),
            quality_fail_on_error=bool(quality.get("fail_on_error", False)),
        )

    def enabled_sources(self) -> tuple[IngestSourceConfig, ...]:
        return tuple(source for source in self.sources if source.enabled)


def load_ingest_config(path: Path) -> IngestPipelineConfig:
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    ingest = payload.get("ingest")
    if not isinstance(ingest, dict):
        raise ValueError("config must contain an [ingest] section")
    return IngestPipelineConfig.from_dict(ingest, base_path=path.parent)


def _source_from_dict(value: dict[str, Any], base_path: Path) -> IngestSourceConfig:
    return IngestSourceConfig(
        name=str(value.get("name", "")),
        tickers=_source_tickers(value, base_path),
        enabled=bool(value.get("enabled", True)),
        start=_optional_string(value.get("start")),
        end=_optional_string(value.get("end")),
        frequency=str(value.get("frequency", "monthly")),
        download_workers=int(value.get("download_workers", 4)),
    )


def _source_tickers(value: dict[str, Any], base_path: Path) -> tuple[str, ...]:
    tickers = [_normalize_ticker(str(ticker)) for ticker in value.get("tickers", [])]
    ticker_file = _optional_string(value.get("ticker_file"))
    if ticker_file:
        ticker_path = Path(ticker_file)
        if not ticker_path.is_absolute():
            ticker_path = base_path / ticker_path
        tickers.extend(_load_ticker_file(ticker_path))
    return tuple(sorted(set(ticker for ticker in tickers if ticker)))


def _load_ticker_file(path: Path) -> list[str]:
    rows: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.split("#", 1)[0].strip()
        if value:
            rows.append(_normalize_ticker(value))
    return rows


def _normalize_ticker(value: str) -> str:
    return value.upper().replace(".", "-")


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
