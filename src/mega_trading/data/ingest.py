"""Ingestion adapters for deterministic and public data sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Generic, TypeVar

from mega_trading.core.schemas import Manifest
from mega_trading.core.store import ArtifactPaths, LocalObjectStore
from mega_trading.data.fixtures import load_fixture_bundle
from mega_trading.data.lance_store import LanceTableStore, LanceTables


@dataclass(frozen=True)
class IngestResult:
    raw_manifest_path: str
    normalization_manifest_path: str
    normalized_counts: dict[str, int]
    quality_summary: dict[str, int]


@dataclass(frozen=True)
class FixtureIngestRequest:
    """Request for deterministic fixture ingestion."""


@dataclass(frozen=True)
class TickerIngestRequest:
    tickers: tuple[str, ...]


@dataclass(frozen=True)
class MarketDataIngestRequest:
    tickers: tuple[str, ...]
    start: str
    end: str


RequestT = TypeVar("RequestT")


class Ingestor(ABC, Generic[RequestT]):
    """Common ingestion contract for config-driven CLIs and schedulers."""

    @abstractmethod
    def ingest(self, request: RequestT) -> IngestResult:
        """Ingest source data and return the written artifact manifests."""


class FixtureIngestor(Ingestor[FixtureIngestRequest]):
    """Load deterministic fixtures into raw, normalized, and quarantine stages."""

    def __init__(self, store: LocalObjectStore, table_store: LanceTableStore | None = None) -> None:
        self.store = store
        self.table_store = table_store
        self.tables = LanceTables()
        self.paths = ArtifactPaths()

    def ingest(self, request: FixtureIngestRequest | None = None) -> IngestResult:
        bundle = load_fixture_bundle()

        raw_paths = {
            "entities": self.paths.raw("fixture", "entities"),
            "documents": self.paths.raw("fixture", "documents"),
            "sec_filings": self.paths.raw("fixture", "sec_filings"),
            "market_data": self.paths.raw("fixture", "market_data"),
        }

        self.store.write_jsonl(raw_paths["entities"], [asdict(row) for row in bundle.entities])
        self.store.write_jsonl(raw_paths["documents"], [asdict(row) for row in bundle.documents])
        self.store.write_jsonl(raw_paths["sec_filings"], [asdict(row) for row in bundle.sec_filings])
        self.store.write_jsonl(raw_paths["market_data"], [asdict(row) for row in bundle.market_data])

        normalized_paths = {
            "entities": self.paths.normalized("entities", "fixture"),
            "documents": self.paths.normalized("documents", "fixture"),
            "sec_filings": self.paths.normalized("sec_filings", "fixture"),
            "market_data": self.paths.normalized("market_data", "fixture"),
        }

        # Fixture records are already normalized typed schemas. Live adapters will
        # use this same normalized contract after source-specific parsing.
        for key, raw_path in raw_paths.items():
            rows = self.store.read_jsonl(raw_path)
            self.store.write_jsonl(normalized_paths[key], rows)
            if self.table_store:
                self.table_store.write_table(self.tables.normalized(key, "fixture"), rows)

        quarantine_path = "quarantine/fixture/bad_records.jsonl"
        self.store.write_jsonl(quarantine_path, bundle.bad_records)

        raw_manifest = Manifest(
            manifest_id="fixture-raw",
            artifact_type="raw",
            paths=list(raw_paths.values()),
            metadata={"source": "fixture", "record_count": str(sum(len(self.store.read_jsonl(path)) for path in raw_paths.values()))},
        )
        raw_manifest_path = self.paths.manifest("ingest", "fixture-raw")
        self.store.write_manifest(raw_manifest_path, raw_manifest)

        normalized_counts = {
            "entities": len(bundle.entities),
            "documents": len(bundle.documents),
            "sec_filings": len(bundle.sec_filings),
            "market_data": len(bundle.market_data),
        }
        quality_summary = {
            "duplicate_records": 0,
            "quarantined_records": len(bundle.bad_records),
        }
        normalization_manifest = Manifest(
            manifest_id="fixture-normalized",
            artifact_type="normalized",
            paths=list(normalized_paths.values()) + [quarantine_path],
            metadata={
                "source_manifest_id": raw_manifest.manifest_id,
                "duplicate_records": str(quality_summary["duplicate_records"]),
                "quarantined_records": str(quality_summary["quarantined_records"]),
            },
        )
        normalization_manifest_path = self.paths.manifest("normalization", "fixture-normalized")
        self.store.write_manifest(normalization_manifest_path, normalization_manifest)

        return IngestResult(
            raw_manifest_path=raw_manifest_path,
            normalization_manifest_path=normalization_manifest_path,
            normalized_counts=normalized_counts,
            quality_summary=quality_summary,
        )
