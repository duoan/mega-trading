"""Configuration contracts used across data, training, eval, and alarms."""

from __future__ import annotations

from dataclasses import dataclass, field

from mega_trading.core.hashing import stable_hash
from mega_trading.core.schemas import SchemaValidationError


@dataclass(frozen=True)
class MixtureSource:
    name: str
    weight: float
    task_type: str

    def __post_init__(self) -> None:
        if not self.name:
            raise SchemaValidationError("mixture source name is required")
        if self.weight <= 0:
            raise SchemaValidationError("mixture source weight must be positive")
        if not self.task_type:
            raise SchemaValidationError("mixture source task_type is required")


@dataclass(frozen=True)
class DataMixtureConfig:
    name: str
    sources: list[MixtureSource]
    description: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise SchemaValidationError("mixture name is required")
        if not self.sources:
            raise SchemaValidationError("mixture sources are required")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "metadata": dict(sorted(self.metadata.items())),
            "sources": [
                {"name": source.name, "weight": source.weight, "task_type": source.task_type}
                for source in self.sources
            ],
        }

    def content_hash(self) -> str:
        return stable_hash(self.to_dict())
