"""Shared artifact schemas for MarketFM Forge.

The schemas intentionally use stdlib dataclasses so the deterministic demo has
no runtime dependency on external validators.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from marketfm.core.hashing import stable_hash


class SchemaValidationError(ValueError):
    """Raised when an artifact violates a schema contract."""


def _require(value: Any, field_name: str) -> None:
    if value is None or value == "" or value == []:
        raise SchemaValidationError(f"{field_name} is required")


def _validate_datetime(value: str, field_name: str) -> None:
    _require(value, field_name)
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SchemaValidationError(f"{field_name} must be ISO-8601: {value}") from exc


def _validate_date(value: str, field_name: str) -> None:
    _require(value, field_name)
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise SchemaValidationError(f"{field_name} must be YYYY-MM-DD: {value}") from exc


@dataclass(frozen=True)
class EntityRecord:
    entity_id: str
    ticker: str
    company_name: str
    cik: str | None = None
    sector: str | None = None
    industry: str | None = None
    source_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _require(self.entity_id, "entity_id")
        _require(self.ticker, "ticker")
        _require(self.company_name, "company_name")


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    entity_id: str
    ticker: str
    source_type: str
    title: str
    text: str
    source_uri: str
    published_at: str
    accepted_at: str
    as_of_time: str
    source_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("document_id", "entity_id", "ticker", "source_type", "text", "source_uri"):
            _require(getattr(self, name), name)
        _validate_datetime(self.published_at, "published_at")
        _validate_datetime(self.accepted_at, "accepted_at")
        _validate_datetime(self.as_of_time, "as_of_time")
        _require(self.source_ids, "source_ids")


@dataclass(frozen=True)
class FundamentalRecord:
    fundamental_id: str
    entity_id: str
    ticker: str
    concept: str
    value: float
    unit: str
    period_end: str
    accepted_at: str
    as_of_time: str
    source_ids: list[str]

    def __post_init__(self) -> None:
        for name in ("fundamental_id", "entity_id", "ticker", "concept", "unit"):
            _require(getattr(self, name), name)
        _validate_date(self.period_end, "period_end")
        _validate_datetime(self.accepted_at, "accepted_at")
        _validate_datetime(self.as_of_time, "as_of_time")
        _require(self.source_ids, "source_ids")


@dataclass(frozen=True)
class PriceRecord:
    price_id: str
    ticker: str
    date: str
    adjusted_close: float
    provider: str
    source_ids: list[str]
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: int | None = None

    def __post_init__(self) -> None:
        for name in ("price_id", "ticker", "provider"):
            _require(getattr(self, name), name)
        _validate_date(self.date, "date")
        _require(self.source_ids, "source_ids")


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    entity_id: str
    ticker: str
    source_type: str
    document_id: str
    timestamp: str
    as_of_time: str
    text: str
    uri: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("evidence_id", "entity_id", "ticker", "source_type", "document_id", "text", "uri"):
            _require(getattr(self, name), name)
        _validate_datetime(self.timestamp, "timestamp")
        _validate_datetime(self.as_of_time, "as_of_time")


@dataclass(frozen=True)
class CorpusRecord:
    corpus_id: str
    task_type: str
    mixture_name: str
    entity_id: str
    ticker: str
    as_of_time: str
    text: str
    source_ids: list[str]
    evidence_ids: list[str]
    quality_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("corpus_id", "task_type", "mixture_name", "entity_id", "ticker", "text"):
            _require(getattr(self, name), name)
        _validate_datetime(self.as_of_time, "as_of_time")
        _require(self.source_ids, "source_ids")
        if not 0.0 <= self.quality_score <= 1.0:
            raise SchemaValidationError("quality_score must be between 0 and 1")


@dataclass(frozen=True)
class Manifest:
    manifest_id: str
    artifact_type: str
    paths: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require(self.manifest_id, "manifest_id")
        _require(self.artifact_type, "artifact_type")
        _require(self.paths, "paths")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def content_hash(self) -> str:
        return stable_hash(self.to_dict())
