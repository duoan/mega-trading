"""Deterministic enrichment over normalized finance records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mega_trading.core.schemas import Manifest
from mega_trading.core.store import ArtifactNotFoundError, LocalObjectStore


@dataclass(frozen=True)
class DataEnrichmentResult:
    snapshot_path: str
    manifest_path: str
    snapshots: int


class DataEnricher:
    def __init__(self, store: LocalObjectStore) -> None:
        self.store = store

    def run(self, run_id: str = "latest") -> DataEnrichmentResult:
        entities = self._read_optional("stage=02_normalized/family=entities/source=sec.jsonl")
        sec_filings = self._read_optional("stage=02_normalized/family=sec_filings/source=sec.jsonl")
        market_data = self._read_optional("stage=02_normalized/family=market_data/source=yahoo.jsonl") + self._read_optional(
            "stage=02_normalized/family=market_data/source=stooq.jsonl"
        )

        sec_filings_by_ticker = _group_by_ticker(sec_filings)
        market_data_by_ticker = _group_by_ticker(market_data)

        snapshots: list[dict[str, Any]] = []
        for entity in entities:
            ticker = str(entity["ticker"])
            ticker_sec_filings = sec_filings_by_ticker.get(ticker, [])
            ticker_market_data = market_data_by_ticker.get(ticker, [])
            latest_sec_filing = max(ticker_sec_filings, key=lambda row: str(row.get("as_of_time", "")), default={})
            market_data_dates = sorted(str(row["date"]) for row in ticker_market_data if row.get("date"))
            snapshots.append(
                {
                    "snapshot_id": f"snapshot-{ticker}",
                    "entity_id": entity["entity_id"],
                    "ticker": ticker,
                    "company_name": entity["company_name"],
                    "latest_sec_filing_as_of_time": latest_sec_filing.get("as_of_time"),
                    "market_data_start": market_data_dates[0] if market_data_dates else None,
                    "market_data_end": market_data_dates[-1] if market_data_dates else None,
                    "market_data_observations": len(ticker_market_data),
                    "source_ids": list(entity.get("source_ids", [])),
                }
            )

        snapshot_path = "stage=03_enriched/company_snapshots.jsonl"
        manifest_path = f"manifests/enrichment/{run_id}.json"
        self.store.write_jsonl(snapshot_path, snapshots)
        self.store.write_manifest(
            manifest_path,
            Manifest(
                manifest_id=f"{run_id}-enrichment",
                artifact_type="enrichment",
                paths=[snapshot_path],
                metadata={"snapshots": str(len(snapshots))},
            ),
        )
        return DataEnrichmentResult(snapshot_path=snapshot_path, manifest_path=manifest_path, snapshots=len(snapshots))

    def _read_optional(self, path: str) -> list[dict[str, Any]]:
        try:
            return self.store.read_jsonl(path)
        except ArtifactNotFoundError:
            return []


def _group_by_ticker(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        ticker = str(row.get("ticker", ""))
        if ticker:
            grouped.setdefault(ticker, []).append(row)
    return grouped
