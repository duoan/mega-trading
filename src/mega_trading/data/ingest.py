"""Ingestion contracts for paper-style order-flow data."""

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
class OhlcvIngestRequest:
    tickers: tuple[str, ...]
    start: str
    end: str


@dataclass(frozen=True)
class BinanceTradesIngestRequest:
    symbols: tuple[str, ...]
    start: str
    end: str
    frequency: str = "monthly"
    download_workers: int = 4


RequestT = TypeVar("RequestT")


class Ingestor(ABC, Generic[RequestT]):
    """Common ingestion contract for config-driven CLIs and schedulers."""

    @abstractmethod
    def ingest(self, request: RequestT) -> IngestResult:
        """Ingest source data and return written artifact manifests."""


class FixtureIngestor(Ingestor[FixtureIngestRequest]):
    """Load deterministic order-flow fixtures into raw and normalized stages."""

    def __init__(self, store: LocalObjectStore, table_store: LanceTableStore | None = None) -> None:
        self.store = store
        self.table_store = table_store
        self.tables = LanceTables()
        self.paths = ArtifactPaths()

    def ingest(self, request: FixtureIngestRequest | None = None) -> IngestResult:
        bundle = load_fixture_bundle()
        raw_path = self.paths.raw("fixture", "order_flow")
        normalized_path = self.paths.normalized("order_flow", "fixture")
        quarantine_path = "quarantine/fixture/bad_records.jsonl"

        rows = [asdict(row) for row in bundle.order_flow]
        self.store.write_jsonl(raw_path, rows)
        self.store.write_jsonl(normalized_path, rows)
        self.store.write_jsonl(quarantine_path, bundle.bad_records)
        if self.table_store:
            self.table_store.write_table(self.tables.normalized("order_flow", "fixture"), rows)

        raw_manifest = Manifest(
            manifest_id="fixture-order-flow-raw",
            artifact_type="raw",
            paths=[raw_path],
            metadata={"source": "fixture", "record_count": str(len(rows))},
        )
        raw_manifest_path = self.paths.manifest("ingest", "fixture-order-flow-raw")
        self.store.write_manifest(raw_manifest_path, raw_manifest)

        quality_summary = {"duplicate_records": 0, "quarantined_records": len(bundle.bad_records)}
        normalization_manifest = Manifest(
            manifest_id="fixture-order-flow-normalized",
            artifact_type="normalized",
            paths=[normalized_path, quarantine_path],
            metadata={
                "source_manifest_id": raw_manifest.manifest_id,
                "order_flow": str(len(rows)),
                "quarantined_records": str(quality_summary["quarantined_records"]),
            },
        )
        normalization_manifest_path = self.paths.manifest("normalization", "fixture-order-flow-normalized")
        self.store.write_manifest(normalization_manifest_path, normalization_manifest)

        return IngestResult(
            raw_manifest_path=raw_manifest_path,
            normalization_manifest_path=normalization_manifest_path,
            normalized_counts={"order_flow": len(rows)},
            quality_summary=quality_summary,
        )
