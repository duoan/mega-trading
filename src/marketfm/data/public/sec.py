"""SEC EDGAR public data adapter."""

from __future__ import annotations

import json
import ssl
from dataclasses import asdict
from typing import Callable
from urllib.request import Request, urlopen

import certifi

from marketfm.core.schemas import EntityRecord, FundamentalRecord, Manifest
from marketfm.core.store import ArtifactPaths, LocalObjectStore
from marketfm.data.ingest import IngestResult
from marketfm.data.lance_store import LanceTableStore, LanceTables

FetchJson = Callable[[str, str], dict]


class SecClient:
    base_url = "https://data.sec.gov"
    ticker_url = "https://www.sec.gov/files/company_tickers.json"

    def __init__(self, user_agent: str, fetch_json: FetchJson | None = None) -> None:
        if not user_agent:
            raise ValueError("SEC user_agent is required")
        self.user_agent = user_agent
        self.fetch_json = fetch_json or _fetch_json

    def resolve_ticker(self, ticker: str) -> dict[str, str]:
        ticker_upper = ticker.upper()
        payload = self.fetch_json(self.ticker_url, self.user_agent)
        for row in payload.values():
            if str(row["ticker"]).upper() == ticker_upper:
                cik = str(row["cik_str"]).zfill(10)
                return {"ticker": ticker_upper, "cik": cik, "company_name": str(row["title"])}
        raise KeyError(f"ticker not found in SEC company_tickers: {ticker}")

    def company_facts(self, ticker: str) -> dict:
        entity = self.resolve_ticker(ticker)
        url = f"{self.base_url}/api/xbrl/companyfacts/CIK{entity['cik']}.json"
        return self.fetch_json(url, self.user_agent)


class SecCompanyFactsIngestor:
    concepts = ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "NetIncomeLoss", "Assets", "Liabilities")

    def __init__(self, store: LocalObjectStore, client: SecClient, table_store: LanceTableStore | None = None) -> None:
        self.store = store
        self.client = client
        self.table_store = table_store
        self.tables = LanceTables()
        self.paths = ArtifactPaths()

    def ingest(self, tickers: list[str]) -> IngestResult:
        raw_rows: list[dict] = []
        entities: list[EntityRecord] = []
        fundamentals: list[FundamentalRecord] = []

        for ticker in tickers:
            entity = self.client.resolve_ticker(ticker)
            facts = self.client.company_facts(ticker)
            raw_rows.append({"ticker": entity["ticker"], "cik": entity["cik"], "payload": facts})
            entities.append(
                EntityRecord(
                    entity_id=f"sec-{entity['cik']}",
                    ticker=entity["ticker"],
                    cik=entity["cik"],
                    company_name=entity["company_name"],
                    source_ids=[f"sec:company_tickers:{entity['cik']}"],
                )
            )
            fundamentals.extend(_fundamentals_from_companyfacts(entity, facts, self.concepts))

        bronze_path = self.paths.bronze("sec", "companyfacts")
        entity_path = self.paths.silver("entities", "sec")
        fundamental_path = self.paths.silver("fundamentals", "sec")
        self.store.write_jsonl(bronze_path, raw_rows)
        entity_rows = [asdict(row) for row in entities]
        fundamental_rows = [asdict(row) for row in fundamentals]
        self.store.write_jsonl(entity_path, entity_rows)
        self.store.write_jsonl(fundamental_path, fundamental_rows)
        if self.table_store:
            self.table_store.write_table(self.tables.silver("entities", "sec"), entity_rows)
            self.table_store.write_table(self.tables.silver("fundamentals", "sec"), fundamental_rows)

        bronze_manifest = Manifest(
            manifest_id="sec-companyfacts-bronze",
            artifact_type="bronze",
            paths=[bronze_path],
            metadata={"source": "sec_companyfacts", "tickers": ",".join(tickers), "record_count": str(len(raw_rows))},
        )
        bronze_manifest_path = self.paths.manifest("ingest", "sec-companyfacts-bronze")
        self.store.write_manifest(bronze_manifest_path, bronze_manifest)

        normalization_manifest = Manifest(
            manifest_id="sec-companyfacts-normalized",
            artifact_type="silver",
            paths=[entity_path, fundamental_path],
            metadata={
                "source": "sec_companyfacts",
                "source_manifest_id": bronze_manifest.manifest_id,
                "entities": str(len(entities)),
                "fundamentals": str(len(fundamentals)),
            },
        )
        normalization_manifest_path = self.paths.manifest("normalization", "sec-companyfacts-normalized")
        self.store.write_manifest(normalization_manifest_path, normalization_manifest)

        return IngestResult(
            bronze_manifest_path=bronze_manifest_path,
            normalization_manifest_path=normalization_manifest_path,
            normalized_counts={"entities": len(entities), "fundamentals": len(fundamentals)},
            quality_summary={"quarantined_records": 0, "duplicate_records": 0},
        )


def _fundamentals_from_companyfacts(entity: dict[str, str], payload: dict, concepts: tuple[str, ...]) -> list[FundamentalRecord]:
    records: list[FundamentalRecord] = []
    gaap = payload.get("facts", {}).get("us-gaap", {})
    for concept in concepts:
        concept_payload = gaap.get(concept)
        if not concept_payload:
            continue
        label = str(concept_payload.get("label", concept))
        for unit, facts in concept_payload.get("units", {}).items():
            for fact in facts:
                if "val" not in fact or "end" not in fact or "filed" not in fact:
                    continue
                filed = str(fact["filed"])
                accepted_at = f"{filed}T00:00:00Z"
                period_end = str(fact["end"])
                form = str(fact.get("form", "unknown"))
                records.append(
                    FundamentalRecord(
                        fundamental_id=f"sec-{entity['ticker']}-{concept}-{period_end}-{filed}",
                        entity_id=f"sec-{entity['cik']}",
                        ticker=entity["ticker"],
                        concept=label,
                        value=float(fact["val"]),
                        unit=str(unit),
                        period_end=period_end,
                        accepted_at=accepted_at,
                        as_of_time=accepted_at,
                        source_ids=[f"sec:companyfacts:{entity['cik']}:{concept}:{period_end}:{form}"],
                    )
                )
    return records


def _fetch_json(url: str, user_agent: str) -> dict:
    request = Request(url, headers={"User-Agent": user_agent, "Accept": "application/json"})
    context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(request, timeout=30, context=context) as response:
        return json.loads(response.read().decode("utf-8"))
