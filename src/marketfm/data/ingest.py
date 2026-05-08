"""Ingestion adapters for deterministic and public data sources."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from marketfm.core.schemas import Manifest
from marketfm.core.store import ArtifactPaths, LocalObjectStore
from marketfm.data.fixtures import load_fixture_bundle
from marketfm.data.lance_store import LanceTableStore, LanceTables


@dataclass(frozen=True)
class IngestResult:
    bronze_manifest_path: str
    normalization_manifest_path: str
    normalized_counts: dict[str, int]
    quality_summary: dict[str, int]


class FixtureIngestor:
    """Load deterministic fixtures into bronze, silver, and quarantine layers."""

    def __init__(self, store: LocalObjectStore, table_store: LanceTableStore | None = None) -> None:
        self.store = store
        self.table_store = table_store
        self.tables = LanceTables()
        self.paths = ArtifactPaths()

    def ingest(self) -> IngestResult:
        bundle = load_fixture_bundle()

        bronze_paths = {
            "entities": self.paths.bronze("fixture", "entities"),
            "documents": self.paths.bronze("fixture", "documents"),
            "fundamentals": self.paths.bronze("fixture", "fundamentals"),
            "prices": self.paths.bronze("fixture", "prices"),
            "evidence": self.paths.bronze("fixture", "evidence"),
            "qa_examples": self.paths.bronze("fixture", "qa_examples"),
            "preference_pairs": self.paths.bronze("fixture", "preference_pairs"),
        }

        self.store.write_jsonl(bronze_paths["entities"], [asdict(row) for row in bundle.entities])
        self.store.write_jsonl(bronze_paths["documents"], [asdict(row) for row in bundle.documents])
        self.store.write_jsonl(bronze_paths["fundamentals"], [asdict(row) for row in bundle.fundamentals])
        self.store.write_jsonl(bronze_paths["prices"], [asdict(row) for row in bundle.prices])
        self.store.write_jsonl(bronze_paths["evidence"], [asdict(row) for row in bundle.evidence])
        self.store.write_jsonl(bronze_paths["qa_examples"], [asdict(row) for row in bundle.qa_examples])
        self.store.write_jsonl(bronze_paths["preference_pairs"], [asdict(row) for row in bundle.preference_pairs])

        silver_paths = {
            "entities": self.paths.silver("entities", "fixture"),
            "documents": self.paths.silver("documents", "fixture"),
            "fundamentals": self.paths.silver("fundamentals", "fixture"),
            "prices": self.paths.silver("prices", "fixture"),
            "evidence": self.paths.silver("evidence", "fixture"),
            "qa_examples": self.paths.silver("qa", "fixture"),
            "preference_pairs": self.paths.silver("preference", "fixture"),
        }

        # Fixture records are already normalized typed schemas. Live adapters will
        # use this same silver contract after source-specific parsing.
        for key, bronze_path in bronze_paths.items():
            rows = self.store.read_jsonl(bronze_path)
            self.store.write_jsonl(silver_paths[key], rows)
            if self.table_store:
                self.table_store.write_table(self.tables.silver(key, "fixture"), rows)

        quarantine_path = "quarantine/fixture/bad_records.jsonl"
        self.store.write_jsonl(quarantine_path, bundle.bad_records)

        bronze_manifest = Manifest(
            manifest_id="fixture-bronze",
            artifact_type="bronze",
            paths=list(bronze_paths.values()),
            metadata={"source": "fixture", "record_count": str(sum(len(self.store.read_jsonl(path)) for path in bronze_paths.values()))},
        )
        bronze_manifest_path = self.paths.manifest("ingest", "fixture-bronze")
        self.store.write_manifest(bronze_manifest_path, bronze_manifest)

        normalized_counts = {
            "entities": len(bundle.entities),
            "documents": len(bundle.documents),
            "fundamentals": len(bundle.fundamentals),
            "prices": len(bundle.prices),
            "evidence": len(bundle.evidence),
            "qa_examples": len(bundle.qa_examples),
            "preference_pairs": len(bundle.preference_pairs),
        }
        quality_summary = {
            "duplicate_records": 0,
            "quarantined_records": len(bundle.bad_records),
        }
        normalization_manifest = Manifest(
            manifest_id="fixture-normalized",
            artifact_type="silver",
            paths=list(silver_paths.values()) + [quarantine_path],
            metadata={
                "source_manifest_id": bronze_manifest.manifest_id,
                "duplicate_records": str(quality_summary["duplicate_records"]),
                "quarantined_records": str(quality_summary["quarantined_records"]),
            },
        )
        normalization_manifest_path = self.paths.manifest("normalization", "fixture-normalized")
        self.store.write_manifest(normalization_manifest_path, normalization_manifest)

        return IngestResult(
            bronze_manifest_path=bronze_manifest_path,
            normalization_manifest_path=normalization_manifest_path,
            normalized_counts=normalized_counts,
            quality_summary=quality_summary,
        )
