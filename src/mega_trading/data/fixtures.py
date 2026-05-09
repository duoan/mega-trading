"""Deterministic fixture data for the local demo path."""

from __future__ import annotations

from dataclasses import dataclass

from mega_trading.core.schemas import (
    DocumentRecord,
    EntityRecord,
    SecFilingRecord,
    MarketDataRecord,
)


@dataclass(frozen=True)
class FixtureBundle:
    entities: list[EntityRecord]
    documents: list[DocumentRecord]
    sec_filings: list[SecFilingRecord]
    market_data: list[MarketDataRecord]
    bad_records: list[dict[str, str]]


def load_fixture_bundle() -> FixtureBundle:
    """Return a small deterministic universe for tests and demos."""

    entities = [
        EntityRecord(
            entity_id="entity-acme",
            ticker="ACME",
            cik="0000000001",
            company_name="Acme Compounders Inc.",
            sector="Industrials",
            industry="Specialty Manufacturing",
            source_ids=["fixture:entities"],
        ),
        EntityRecord(
            entity_id="entity-nova",
            ticker="NOVA",
            cik="0000000002",
            company_name="Nova Retail Group",
            sector="Consumer Discretionary",
            industry="Specialty Retail",
            source_ids=["fixture:entities"],
        ),
    ]

    documents = [
        DocumentRecord(
            document_id="doc-acme-2022-10k",
            entity_id="entity-acme",
            ticker="ACME",
            source_type="10-K",
            title="ACME 2022 Form 10-K",
            text=(
                "ACME expanded recurring service revenue and improved operating margin. "
                "Management noted supply chain risk and customer concentration."
            ),
            source_uri="fixture://sec_filings/acme-2022-10k",
            published_at="2023-02-15T16:30:00Z",
            accepted_at="2023-02-15T16:30:00Z",
            as_of_time="2023-02-15T16:30:00Z",
            source_ids=["raw:acme-10k"],
        ),
        DocumentRecord(
            document_id="doc-nova-2022-10k",
            entity_id="entity-nova",
            ticker="NOVA",
            source_type="10-K",
            title="NOVA 2022 Form 10-K",
            text=(
                "NOVA reported slowing same-store sales and higher inventory markdowns. "
                "Management highlighted debt reduction as a priority."
            ),
            source_uri="fixture://sec_filings/nova-2022-10k",
            published_at="2023-03-01T16:30:00Z",
            accepted_at="2023-03-01T16:30:00Z",
            as_of_time="2023-03-01T16:30:00Z",
            source_ids=["raw:nova-10k"],
        ),
    ]

    sec_filings = [
        SecFilingRecord(
            sec_filing_id="fact-acme-revenue-2022",
            entity_id="entity-acme",
            ticker="ACME",
            concept="Revenue",
            value=1200.0,
            unit="USDm",
            period_end="2022-12-31",
            accepted_at="2023-02-15T16:30:00Z",
            as_of_time="2023-02-15T16:30:00Z",
            source_ids=["raw:acme-10k"],
        ),
        SecFilingRecord(
            sec_filing_id="fact-acme-net-income-2022",
            entity_id="entity-acme",
            ticker="ACME",
            concept="NetIncome",
            value=180.0,
            unit="USDm",
            period_end="2022-12-31",
            accepted_at="2023-02-15T16:30:00Z",
            as_of_time="2023-02-15T16:30:00Z",
            source_ids=["raw:acme-10k"],
        ),
        SecFilingRecord(
            sec_filing_id="fact-nova-revenue-2022",
            entity_id="entity-nova",
            ticker="NOVA",
            concept="Revenue",
            value=950.0,
            unit="USDm",
            period_end="2022-12-31",
            accepted_at="2023-03-01T16:30:00Z",
            as_of_time="2023-03-01T16:30:00Z",
            source_ids=["raw:nova-10k"],
        ),
        SecFilingRecord(
            sec_filing_id="fact-nova-net-income-2022",
            entity_id="entity-nova",
            ticker="NOVA",
            concept="NetIncome",
            value=25.0,
            unit="USDm",
            period_end="2022-12-31",
            accepted_at="2023-03-01T16:30:00Z",
            as_of_time="2023-03-01T16:30:00Z",
            source_ids=["raw:nova-10k"],
        ),
    ]

    market_data = [
        MarketDataRecord("px-acme-2023-01-31", "ACME", "2023-01-31", 40.0, "fixture", ["raw:market_data"]),
        MarketDataRecord("px-acme-2023-02-28", "ACME", "2023-02-28", 42.0, "fixture", ["raw:market_data"]),
        MarketDataRecord("px-acme-2023-08-31", "ACME", "2023-08-31", 48.0, "fixture", ["raw:market_data"]),
        MarketDataRecord("px-acme-2024-02-29", "ACME", "2024-02-29", 55.0, "fixture", ["raw:market_data"]),
        MarketDataRecord("px-nova-2023-01-31", "NOVA", "2023-01-31", 30.0, "fixture", ["raw:market_data"]),
        MarketDataRecord("px-nova-2023-03-31", "NOVA", "2023-03-31", 28.0, "fixture", ["raw:market_data"]),
        MarketDataRecord("px-nova-2023-09-29", "NOVA", "2023-09-29", 24.0, "fixture", ["raw:market_data"]),
        MarketDataRecord("px-nova-2024-03-29", "NOVA", "2024-03-29", 22.0, "fixture", ["raw:market_data"]),
    ]

    bad_records = [
        {"record_id": "bad-stale-news", "reason": "stale_feed"},
        {"record_id": "bad-future-feature", "reason": "future_leakage"},
        {"record_id": "bad-missing-entity", "reason": "missing_entity"},
    ]

    return FixtureBundle(
        entities=entities,
        documents=documents,
        sec_filings=sec_filings,
        market_data=market_data,
        bad_records=bad_records,
    )
