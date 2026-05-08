"""Config-driven ingestion contracts."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class IngestSourceConfig:
    name: str
    tickers: tuple[str, ...]
    enabled: bool = True
    start: str | None = None
    end: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ingest source name is required")
        if not self.tickers:
            raise ValueError(f"ingest source {self.name} requires tickers")
        if self.name in {"yahoo_prices", "stooq_prices"} and (not self.start or not self.end):
            raise ValueError(f"price source {self.name} requires start and end")


@dataclass(frozen=True)
class IngestPipelineConfig:
    output_dir: str
    sources: tuple[IngestSourceConfig, ...]
    sec_user_agent: str | None = None
    quality_enabled: bool = True
    quality_fail_on_error: bool = False
    enrichment_enabled: bool = True
    training_data_enabled: bool = True
    training_mixture_name: str = "public"
    training_sequence_length: int = 32
    training_input_window_observations: int = 20
    training_horizon_observations: int = 20
    training_return_threshold: float = 0.02

    def __post_init__(self) -> None:
        if not self.output_dir:
            raise ValueError("ingest output_dir is required")
        if not self.sources:
            raise ValueError("at least one ingest source is required")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "IngestPipelineConfig":
        sources = tuple(_source_from_dict(source) for source in value.get("sources", []))
        quality = value.get("quality", {})
        if not isinstance(quality, dict):
            quality = {}
        enrichment = value.get("enrichment", {})
        if not isinstance(enrichment, dict):
            enrichment = {}
        training_data = value.get("training_data", {})
        if not isinstance(training_data, dict):
            training_data = {}
        return cls(
            output_dir=str(value.get("output_dir", "")),
            sec_user_agent=_optional_string(value.get("sec_user_agent")),
            sources=sources,
            quality_enabled=bool(quality.get("enabled", True)),
            quality_fail_on_error=bool(quality.get("fail_on_error", False)),
            enrichment_enabled=bool(enrichment.get("enabled", True)),
            training_data_enabled=bool(training_data.get("enabled", True)),
            training_mixture_name=str(training_data.get("mixture_name", "public")),
            training_sequence_length=int(training_data.get("sequence_length", 32)),
            training_input_window_observations=int(training_data.get("input_window_observations", 20)),
            training_horizon_observations=int(training_data.get("horizon_observations", 20)),
            training_return_threshold=float(training_data.get("return_threshold", 0.02)),
        )

    def enabled_sources(self) -> tuple[IngestSourceConfig, ...]:
        return tuple(source for source in self.sources if source.enabled)


def load_ingest_config(path: Path) -> IngestPipelineConfig:
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    ingest = payload.get("ingest")
    if not isinstance(ingest, dict):
        raise ValueError("config must contain an [ingest] section")
    return IngestPipelineConfig.from_dict(ingest)


def _source_from_dict(value: dict[str, Any]) -> IngestSourceConfig:
    return IngestSourceConfig(
        name=str(value.get("name", "")),
        tickers=tuple(str(ticker).upper() for ticker in value.get("tickers", [])),
        enabled=bool(value.get("enabled", True)),
        start=_optional_string(value.get("start")),
        end=_optional_string(value.get("end")),
    )


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
