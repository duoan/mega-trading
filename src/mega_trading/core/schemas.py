"""Shared artifact schemas for Mega-Trading.

The schemas intentionally use stdlib dataclasses so the deterministic demo has
no runtime dependency on external validators.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from mega_trading.core.hashing import stable_hash


class SchemaValidationError(ValueError):
    """Raised when an artifact violates a schema contract."""


def _require(value: Any, field_name: str) -> None:
    if value is None or value == "" or value == [] or value == {}:
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
class SecFilingRecord:
    sec_filing_id: str
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
        for name in ("sec_filing_id", "entity_id", "ticker", "concept", "unit"):
            _require(getattr(self, name), name)
        _validate_date(self.period_end, "period_end")
        _validate_datetime(self.accepted_at, "accepted_at")
        _validate_datetime(self.as_of_time, "as_of_time")
        _require(self.source_ids, "source_ids")


@dataclass(frozen=True)
class MarketDataRecord:
    market_data_id: str
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
        for name in ("market_data_id", "ticker", "provider"):
            _require(getattr(self, name), name)
        _validate_date(self.date, "date")
        _require(self.source_ids, "source_ids")


@dataclass(frozen=True)
class ModelVersionRecord:
    model_version_id: str
    run_id: str
    checkpoint_path: str
    manifest_path: str
    metrics_path: str
    base_model_version: str
    adapter_version: str
    head_version: str
    feature_version: str
    label_version: str
    data_snapshot_version: str
    config_hash: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "model_version_id",
            "run_id",
            "checkpoint_path",
            "manifest_path",
            "metrics_path",
            "base_model_version",
            "adapter_version",
            "head_version",
            "feature_version",
            "label_version",
            "data_snapshot_version",
            "config_hash",
        ):
            _require(getattr(self, name), name)
        _validate_datetime(self.created_at, "created_at")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def content_hash(self) -> str:
        return stable_hash(self.to_dict())


@dataclass(frozen=True)
class PredictionRecord:
    prediction_id: str
    prediction_time: str
    ticker: str
    sample_id: str
    model_version_id: str
    base_model_version: str
    adapter_version: str
    head_version: str
    feature_version: str
    label_version: str
    pred_return_bucket: str
    pred_risk_bucket: str
    confidence: float
    return_probabilities: dict[str, float]
    risk_probabilities: dict[str, float]
    source_ids: list[str]
    label_status: str = "pending"
    actual_return_bucket: str | None = None
    actual_risk_bucket: str | None = None
    actual_forward_return: float | None = None
    label_ready_time: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "prediction_id",
            "prediction_time",
            "ticker",
            "sample_id",
            "model_version_id",
            "base_model_version",
            "adapter_version",
            "head_version",
            "feature_version",
            "label_version",
            "pred_return_bucket",
            "pred_risk_bucket",
            "label_status",
        ):
            _require(getattr(self, name), name)
        _validate_datetime(self.prediction_time, "prediction_time")
        if self.label_ready_time is not None:
            _validate_datetime(self.label_ready_time, "label_ready_time")
        if not 0.0 <= self.confidence <= 1.0:
            raise SchemaValidationError("confidence must be in [0.0, 1.0]")
        _require(self.return_probabilities, "return_probabilities")
        _require(self.risk_probabilities, "risk_probabilities")
        _require(self.source_ids, "source_ids")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DelayedLabelRecord:
    label_id: str
    prediction_id: str
    ticker: str
    prediction_time: str
    label_ready_time: str
    actual_return_bucket: str
    actual_risk_bucket: str
    actual_forward_return: float
    source_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "label_id",
            "prediction_id",
            "ticker",
            "actual_return_bucket",
            "actual_risk_bucket",
        ):
            _require(getattr(self, name), name)
        _validate_datetime(self.prediction_time, "prediction_time")
        _validate_datetime(self.label_ready_time, "label_ready_time")
        _require(self.source_ids, "source_ids")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BacktestResultRecord:
    backtest_id: str
    created_at: str
    prediction_path: str
    labeled_prediction_path: str
    metrics: dict[str, float]
    config: dict[str, Any]
    source_ids: list[str]

    def __post_init__(self) -> None:
        for name in ("backtest_id", "prediction_path", "labeled_prediction_path"):
            _require(getattr(self, name), name)
        _validate_datetime(self.created_at, "created_at")
        _require(self.metrics, "metrics")
        _require(self.source_ids, "source_ids")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
