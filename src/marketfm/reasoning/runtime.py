"""Evidence-grounded reasoning runtime for long-term investment theses."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from marketfm.core.store import ArtifactPaths, LocalObjectStore
from marketfm.data.lance_store import LanceTableStore, LanceTables


class ReasoningValidationError(ValueError):
    """Raised when a reasoning output violates the output contract."""


@dataclass(frozen=True)
class EvidencePack:
    ticker: str
    as_of_time: str
    horizon: str
    prompt: str
    evidence_ids: list[str]
    evidence: list[dict[str, Any]]


class EvidenceCatalog:
    def __init__(self, store: LocalObjectStore) -> None:
        self.store = store

    def query(self, ticker: str, as_of_time: str, limit: int = 5) -> list[dict[str, Any]]:
        boundary = _parse_time(as_of_time)
        rows = self.store.read_jsonl("stage=02_normalized/family=evidence/source=fixture.jsonl")
        candidates = [
            row
            for row in rows
            if row["ticker"] == ticker and _parse_time(str(row["timestamp"])) <= boundary
        ]
        return candidates[:limit]


class LanceEvidenceCatalog:
    def __init__(self, table_store: LanceTableStore, source: str = "fixture") -> None:
        self.table_store = table_store
        self.table_name = LanceTables().normalized("evidence", source)

    def query(self, ticker: str, as_of_time: str, limit: int = 5) -> list[dict[str, Any]]:
        boundary = _parse_time(as_of_time)
        rows = self.table_store.read_rows(self.table_name)
        candidates = [
            row
            for row in rows
            if row["ticker"] == ticker and _parse_time(str(row["timestamp"])) <= boundary
        ]
        return candidates[:limit]


class EvidencePackBuilder:
    def __init__(self, catalog: EvidenceCatalog | LanceEvidenceCatalog) -> None:
        self.catalog = catalog

    def build(self, ticker: str, as_of_time: str, horizon: str) -> EvidencePack:
        evidence = self.catalog.query(ticker, as_of_time)
        lines = [
            f"Task: produce a long-term investment thesis as of {as_of_time}.",
            f"Ticker: {ticker}",
            f"Horizon: {horizon}",
            "Evidence:",
        ]
        for row in evidence:
            lines.append(f"[{row['evidence_id']}] {row['text']}")
        lines.append("Use only the evidence above and cite evidence IDs.")
        return EvidencePack(
            ticker=ticker,
            as_of_time=as_of_time,
            horizon=horizon,
            prompt="\n".join(lines),
            evidence_ids=[str(row["evidence_id"]) for row in evidence],
            evidence=evidence,
        )


class ReasoningRuntime:
    def __init__(self, store: LocalObjectStore, run_id: str = "demo", table_store: LanceTableStore | None = None) -> None:
        self.store = store
        self.table_store = table_store
        self.paths = ArtifactPaths(run_id=run_id)

    def reason(self, ticker: str, as_of_time: str, horizon: str) -> dict[str, Any]:
        catalog = LanceEvidenceCatalog(self.table_store) if self.table_store else EvidenceCatalog(self.store)
        pack = EvidencePackBuilder(catalog).build(ticker, as_of_time, horizon)
        if not pack.evidence:
            output = self._insufficient_evidence(ticker, as_of_time, horizon)
        else:
            output = self._thesis_from_pack(pack)
        self.validate(output)
        self.store.write_json(self.paths.run("reasoning-output.json"), output)
        return output

    def validate(self, output: dict[str, Any]) -> None:
        required = ["ticker", "as_of_time", "horizon", "investment_view", "confidence", "thesis", "evidence", "lineage"]
        for field in required:
            if field not in output:
                raise ReasoningValidationError(f"missing field: {field}")
        if output["investment_view"] != "insufficient_evidence" and not output["evidence"]:
            raise ReasoningValidationError("non-empty evidence is required for investment views")
        if not 0.0 <= float(output["confidence"]) <= 1.0:
            raise ReasoningValidationError("confidence must be between 0 and 1")

    def _insufficient_evidence(self, ticker: str, as_of_time: str, horizon: str) -> dict[str, Any]:
        return {
            "ticker": ticker,
            "as_of_time": as_of_time,
            "horizon": horizon,
            "investment_view": "insufficient_evidence",
            "ranking_score": 0.0,
            "confidence": 0.0,
            "thesis": "Insufficient evidence is available before the requested as_of_time.",
            "business_quality": {"summary": "Not assessed.", "evidence_ids": []},
            "valuation": {"summary": "Not assessed.", "evidence_ids": []},
            "catalysts": [],
            "risks": [],
            "uncertainties": ["No valid evidence was available before as_of_time."],
            "evidence": [],
            "lineage": _lineage([], "investment-thesis-v1"),
        }

    def _thesis_from_pack(self, pack: EvidencePack) -> dict[str, Any]:
        risk_evidence = [row for row in pack.evidence if "risk" in str(row["text"]).lower() or "slowing" in str(row["text"]).lower()]
        investment_view = "neutral" if risk_evidence else "attractive"
        confidence = min(0.8, 0.4 + 0.1 * len(pack.evidence))
        thesis = f"{pack.ticker} has an evidence-grounded {investment_view} long-term setup based on cited company disclosures."
        return {
            "ticker": pack.ticker,
            "as_of_time": pack.as_of_time,
            "horizon": pack.horizon,
            "investment_view": investment_view,
            "ranking_score": confidence,
            "confidence": confidence,
            "thesis": thesis,
            "business_quality": {"summary": "Model-visible evidence indicates durable operating signals.", "evidence_ids": pack.evidence_ids[:1]},
            "valuation": {"summary": "Valuation is not fully assessed in the fixture runtime.", "evidence_ids": []},
            "catalysts": [],
            "risks": [{"claim": "Risk evidence is present and should be monitored.", "evidence_ids": [row["evidence_id"] for row in risk_evidence]}],
            "uncertainties": ["Fixture runtime uses limited public evidence."],
            "evidence": [
                {
                    "id": row["evidence_id"],
                    "source_type": row["source_type"],
                    "timestamp": row["timestamp"],
                    "snippet": row["text"],
                    "uri": row["uri"],
                }
                for row in pack.evidence
            ],
            "lineage": _lineage(pack.evidence_ids, "investment-thesis-v1"),
        }


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _lineage(evidence_ids: list[str], prompt_version: str) -> dict[str, Any]:
    return {
        "model_version": "fixture-reasoner-v1",
        "data_mixture_version": "fixture",
        "evidence_ids": evidence_ids,
        "prompt_version": prompt_version,
    }
