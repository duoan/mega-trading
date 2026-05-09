"""Shared artifact schemas for Mega-Trading."""

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
class OrderFlowEventRecord:
    """Paper-style participant-visible order-flow event features."""

    event_id: str
    ticker: str
    timestamp: str
    date: str
    action: str
    side: str
    midprice: float
    relative_price_bps: float
    price_depth_bps: float
    size: float
    interarrival_seconds: float
    provider: str
    source_ids: list[str]
    midprice_return_bps: float | None = None

    def __post_init__(self) -> None:
        for name in ("event_id", "ticker", "timestamp", "date", "action", "side", "provider"):
            _require(getattr(self, name), name)
        if self.action not in {"add", "delete"}:
            raise SchemaValidationError(f"action must be add or delete: {self.action}")
        if self.side not in {"buy", "sell"}:
            raise SchemaValidationError(f"side must be buy or sell: {self.side}")
        if self.midprice <= 0.0:
            raise SchemaValidationError("midprice must be positive")
        if self.price_depth_bps < 0.0:
            raise SchemaValidationError("price_depth_bps must be non-negative")
        if self.size <= 0.0:
            raise SchemaValidationError("size must be positive")
        if self.interarrival_seconds <= 0.0:
            raise SchemaValidationError("interarrival_seconds must be positive")
        _validate_datetime(self.timestamp, "timestamp")
        _validate_date(self.date, "date")
        _require(self.source_ids, "source_ids")


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
