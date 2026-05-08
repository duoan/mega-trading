"""Deterministic fixture data for the local demo path."""

from __future__ import annotations

from dataclasses import dataclass

from marketfm.schemas import (
    DocumentRecord,
    EntityRecord,
    EvidenceRecord,
    FundamentalRecord,
    PriceRecord,
)


@dataclass(frozen=True)
class QAExample:
    example_id: str
    question: str
    answer: str
    evidence_ids: list[str]
    as_of_time: str


@dataclass(frozen=True)
class PreferencePair:
    pair_id: str
    prompt: str
    chosen: str
    rejected: str
    evidence_ids: list[str]
    as_of_time: str


@dataclass(frozen=True)
class FixtureBundle:
    entities: list[EntityRecord]
    documents: list[DocumentRecord]
    fundamentals: list[FundamentalRecord]
    prices: list[PriceRecord]
    evidence: list[EvidenceRecord]
    qa_examples: list[QAExample]
    preference_pairs: list[PreferencePair]
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
            source_uri="fixture://filings/acme-2022-10k",
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
            source_uri="fixture://filings/nova-2022-10k",
            published_at="2023-03-01T16:30:00Z",
            accepted_at="2023-03-01T16:30:00Z",
            as_of_time="2023-03-01T16:30:00Z",
            source_ids=["raw:nova-10k"],
        ),
    ]

    fundamentals = [
        FundamentalRecord(
            fundamental_id="fact-acme-revenue-2022",
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
        FundamentalRecord(
            fundamental_id="fact-acme-net-income-2022",
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
        FundamentalRecord(
            fundamental_id="fact-nova-revenue-2022",
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
        FundamentalRecord(
            fundamental_id="fact-nova-net-income-2022",
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

    prices = [
        PriceRecord("px-acme-2023-01-31", "ACME", "2023-01-31", 40.0, "fixture", ["raw:prices"]),
        PriceRecord("px-acme-2023-02-28", "ACME", "2023-02-28", 42.0, "fixture", ["raw:prices"]),
        PriceRecord("px-acme-2023-08-31", "ACME", "2023-08-31", 48.0, "fixture", ["raw:prices"]),
        PriceRecord("px-acme-2024-02-29", "ACME", "2024-02-29", 55.0, "fixture", ["raw:prices"]),
        PriceRecord("px-nova-2023-01-31", "NOVA", "2023-01-31", 30.0, "fixture", ["raw:prices"]),
        PriceRecord("px-nova-2023-03-31", "NOVA", "2023-03-31", 28.0, "fixture", ["raw:prices"]),
        PriceRecord("px-nova-2023-09-29", "NOVA", "2023-09-29", 24.0, "fixture", ["raw:prices"]),
        PriceRecord("px-nova-2024-03-29", "NOVA", "2024-03-29", 22.0, "fixture", ["raw:prices"]),
    ]

    evidence = [
        EvidenceRecord(
            evidence_id="ev-acme-margin",
            entity_id="entity-acme",
            ticker="ACME",
            source_type="10-K",
            document_id="doc-acme-2022-10k",
            timestamp="2023-02-15T16:30:00Z",
            as_of_time="2023-02-15T16:30:00Z",
            text="ACME improved operating margin while expanding recurring service revenue.",
            uri="fixture://filings/acme-2022-10k#item7",
        ),
        EvidenceRecord(
            evidence_id="ev-acme-risk",
            entity_id="entity-acme",
            ticker="ACME",
            source_type="10-K",
            document_id="doc-acme-2022-10k",
            timestamp="2023-02-15T16:30:00Z",
            as_of_time="2023-02-15T16:30:00Z",
            text="Management noted supply chain risk and customer concentration.",
            uri="fixture://filings/acme-2022-10k#risk",
        ),
        EvidenceRecord(
            evidence_id="ev-nova-sales",
            entity_id="entity-nova",
            ticker="NOVA",
            source_type="10-K",
            document_id="doc-nova-2022-10k",
            timestamp="2023-03-01T16:30:00Z",
            as_of_time="2023-03-01T16:30:00Z",
            text="NOVA reported slowing same-store sales and higher inventory markdowns.",
            uri="fixture://filings/nova-2022-10k#item7",
        ),
        EvidenceRecord(
            evidence_id="ev-nova-debt",
            entity_id="entity-nova",
            ticker="NOVA",
            source_type="10-K",
            document_id="doc-nova-2022-10k",
            timestamp="2023-03-01T16:30:00Z",
            as_of_time="2023-03-01T16:30:00Z",
            text="Management highlighted debt reduction as a priority.",
            uri="fixture://filings/nova-2022-10k#liquidity",
        ),
    ]

    qa_examples = [
        QAExample(
            example_id="qa-acme-margin",
            question="What evidence supports ACME's business quality as of 2023-02-15?",
            answer="ACME expanded recurring service revenue and improved operating margin.",
            evidence_ids=["ev-acme-margin"],
            as_of_time="2023-02-15T16:30:00Z",
        ),
        QAExample(
            example_id="qa-nova-risk",
            question="What evidence weakens NOVA's investment thesis as of 2023-03-01?",
            answer="NOVA reported slowing same-store sales and higher inventory markdowns.",
            evidence_ids=["ev-nova-sales"],
            as_of_time="2023-03-01T16:30:00Z",
        ),
    ]

    preference_pairs = [
        PreferencePair(
            pair_id="pref-acme-cited",
            prompt="Write an ACME investment thesis using the provided evidence.",
            chosen="ACME looks attractive because recurring revenue and margins improved [ev-acme-margin].",
            rejected="ACME will definitely outperform because it is a great company.",
            evidence_ids=["ev-acme-margin"],
            as_of_time="2023-02-15T16:30:00Z",
        ),
        PreferencePair(
            pair_id="pref-nova-cautious",
            prompt="Write a NOVA investment thesis using the provided evidence.",
            chosen="NOVA requires caution because sales slowed and markdowns increased [ev-nova-sales].",
            rejected="NOVA is a strong buy because its future turnaround is guaranteed.",
            evidence_ids=["ev-nova-sales"],
            as_of_time="2023-03-01T16:30:00Z",
        ),
    ]

    bad_records = [
        {"record_id": "bad-stale-news", "reason": "stale_feed"},
        {"record_id": "bad-future-evidence", "reason": "future_leakage"},
        {"record_id": "bad-missing-entity", "reason": "missing_entity"},
    ]

    return FixtureBundle(
        entities=entities,
        documents=documents,
        fundamentals=fundamentals,
        prices=prices,
        evidence=evidence,
        qa_examples=qa_examples,
        preference_pairs=preference_pairs,
        bad_records=bad_records,
    )
