"""Build model-facing corpora from normalized financial records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json

from marketfm.core.config import DataMixtureConfig
from marketfm.core.schemas import CorpusRecord, Manifest
from marketfm.core.store import ArtifactNotFoundError, ArtifactPaths, LocalObjectStore


class LeakageError(ValueError):
    """Raised when future evidence enters a model-visible artifact."""


class ReadinessError(ValueError):
    """Raised when public data is not marked ready for training."""


@dataclass(frozen=True)
class CorpusBuildResult:
    manifest_path: str
    counts: dict[str, int]


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class CorpusBuilder:
    def __init__(self, store: LocalObjectStore) -> None:
        self.store = store
        self.paths = ArtifactPaths()

    def build(self, mixture: DataMixtureConfig) -> CorpusBuildResult:
        documents = self.store.read_jsonl("stage=02_normalized/family=documents/source=fixture.jsonl")
        qa_examples = self.store.read_jsonl("stage=02_normalized/family=qa/source=fixture.jsonl")
        preference_pairs = self.store.read_jsonl("stage=02_normalized/family=preference/source=fixture.jsonl")
        evidence = {
            row["evidence_id"]: row
            for row in self.store.read_jsonl("stage=02_normalized/family=evidence/source=fixture.jsonl")
        }

        cpt_records = self._build_cpt(mixture.name, documents)
        sft_records = self._build_sft(mixture.name, qa_examples, evidence)
        preference_records = self._build_preference(mixture.name, preference_pairs, evidence)

        cpt_path = self.paths.corpus(mixture.name, "cpt")
        sft_path = self.paths.corpus(mixture.name, "sft")
        preference_path = self.paths.corpus(mixture.name, "preference")

        self.store.write_jsonl(cpt_path, [asdict(row) for row in cpt_records])
        self.store.write_jsonl(sft_path, [asdict(row) for row in sft_records])
        self.store.write_jsonl(preference_path, [asdict(row) for row in preference_records])

        manifest = Manifest(
            manifest_id=f"{mixture.name}-corpus",
            artifact_type="corpus",
            paths=[cpt_path, sft_path, preference_path],
            metadata={
                "mixture_name": mixture.name,
                "mixture_hash": mixture.content_hash(),
                "cpt_records": str(len(cpt_records)),
                "sft_records": str(len(sft_records)),
                "preference_records": str(len(preference_records)),
            },
        )
        manifest_path = self.paths.manifest("corpus", f"{mixture.name}-corpus")
        self.store.write_manifest(manifest_path, manifest)

        return CorpusBuildResult(
            manifest_path=manifest_path,
            counts={"cpt": len(cpt_records), "sft": len(sft_records), "preference": len(preference_records)},
        )

    def _build_cpt(self, mixture_name: str, documents: list[dict[str, object]]) -> list[CorpusRecord]:
        records: list[CorpusRecord] = []
        for document in documents:
            document_id = str(document["document_id"])
            records.append(
                CorpusRecord(
                    corpus_id=f"cpt-{document_id}",
                    task_type="cpt_text",
                    mixture_name=mixture_name,
                    entity_id=str(document["entity_id"]),
                    ticker=str(document["ticker"]),
                    as_of_time=str(document["as_of_time"]),
                    text=str(document["text"]),
                    source_ids=[document_id],
                    evidence_ids=[],
                    quality_score=1.0,
                )
            )
        return records

    def _build_sft(
        self,
        mixture_name: str,
        qa_examples: list[dict[str, object]],
        evidence: dict[str, dict[str, object]],
    ) -> list[CorpusRecord]:
        records: list[CorpusRecord] = []
        for example in qa_examples:
            evidence_ids = [str(evidence_id) for evidence_id in example["evidence_ids"]]
            self._ensure_evidence_not_future(evidence_ids, str(example["as_of_time"]), evidence)
            source_ids = [str(evidence[evidence_id]["document_id"]) for evidence_id in evidence_ids]
            text = (
                f"Question: {example['question']}\n"
                f"Answer: {example['answer']}\n"
                f"Evidence: {', '.join(evidence_ids)}"
            )
            records.append(
                CorpusRecord(
                    corpus_id=f"sft-{example['example_id']}",
                    task_type="sft_instruction",
                    mixture_name=mixture_name,
                    entity_id=str(evidence[evidence_ids[0]]["entity_id"]),
                    ticker=str(evidence[evidence_ids[0]]["ticker"]),
                    as_of_time=str(example["as_of_time"]),
                    text=text,
                    source_ids=source_ids,
                    evidence_ids=evidence_ids,
                    quality_score=1.0,
                )
            )
        return records

    def _build_preference(
        self,
        mixture_name: str,
        preference_pairs: list[dict[str, object]],
        evidence: dict[str, dict[str, object]],
    ) -> list[CorpusRecord]:
        records: list[CorpusRecord] = []
        for pair in preference_pairs:
            evidence_ids = [str(evidence_id) for evidence_id in pair["evidence_ids"]]
            self._ensure_evidence_not_future(evidence_ids, str(pair["as_of_time"]), evidence)
            first_evidence = evidence[evidence_ids[0]]
            for label in ("chosen", "rejected"):
                records.append(
                    CorpusRecord(
                        corpus_id=f"pref-{pair['pair_id']}-{label}",
                        task_type=f"preference_{label}",
                        mixture_name=mixture_name,
                        entity_id=str(first_evidence["entity_id"]),
                        ticker=str(first_evidence["ticker"]),
                        as_of_time=str(pair["as_of_time"]),
                        text=f"Prompt: {pair['prompt']}\nResponse: {pair[label]}",
                        source_ids=[str(first_evidence["document_id"])],
                        evidence_ids=evidence_ids,
                        quality_score=1.0 if label == "chosen" else 0.5,
                    )
                )
        return records

    def _ensure_evidence_not_future(
        self,
        evidence_ids: list[str],
        as_of_time: str,
        evidence: dict[str, dict[str, object]],
    ) -> None:
        boundary = _parse_time(as_of_time)
        for evidence_id in evidence_ids:
            evidence_time = _parse_time(str(evidence[evidence_id]["timestamp"]))
            if evidence_time > boundary:
                raise LeakageError(f"evidence {evidence_id} is after as_of_time {as_of_time}")


class PublicCorpusBuilder:
    """Build trainable corpus records from readiness-passed public data."""

    def __init__(self, store: LocalObjectStore) -> None:
        self.store = store
        self.paths = ArtifactPaths()

    def build(self, mixture_name: str = "public") -> CorpusBuildResult:
        readiness = self.store.read_json("reports/data-readiness.json")
        if not readiness.get("training_ready"):
            raise ReadinessError("public data readiness report is not training_ready")

        snapshots = self.store.read_jsonl("stage=03_enriched/company_snapshots.jsonl")
        fundamentals = _group_by_ticker(self._read_optional("stage=02_normalized/family=fundamentals/source=sec.jsonl"))
        prices = _group_by_ticker(
            self._read_optional("stage=02_normalized/family=prices/source=yahoo.jsonl")
            + self._read_optional("stage=02_normalized/family=prices/source=stooq.jsonl")
        )
        cpt_records = [
            self._snapshot_to_cpt(
                mixture_name,
                snapshot,
                fundamentals.get(str(snapshot["ticker"]), []),
                prices.get(str(snapshot["ticker"]), []),
            )
            for snapshot in snapshots
        ]
        sft_records = [
            self._snapshot_to_sft(
                mixture_name,
                snapshot,
                fundamentals.get(str(snapshot["ticker"]), []),
                prices.get(str(snapshot["ticker"]), []),
            )
            for snapshot in snapshots
        ]
        cpt_path = self.paths.corpus(mixture_name, "cpt")
        sft_path = self.paths.corpus(mixture_name, "sft")
        self.store.write_jsonl(cpt_path, [asdict(record) for record in cpt_records])
        self.store.write_jsonl(sft_path, [asdict(record) for record in sft_records])

        manifest = Manifest(
            manifest_id=f"{mixture_name}-public-corpus",
            artifact_type="corpus",
            paths=[cpt_path, sft_path],
            metadata={
                "mixture_name": mixture_name,
                "source": "public_readiness_enriched_snapshots_fundamentals_prices",
                "cpt_records": str(len(cpt_records)),
                "sft_records": str(len(sft_records)),
                "readiness_quality_score": str(readiness.get("quality_score", "")),
            },
        )
        manifest_path = self.paths.manifest("corpus", f"{mixture_name}-public-corpus")
        self.store.write_manifest(manifest_path, manifest)
        return CorpusBuildResult(manifest_path=manifest_path, counts={"cpt": len(cpt_records), "sft": len(sft_records)})

    def _snapshot_to_cpt(
        self,
        mixture_name: str,
        snapshot: dict[str, object],
        fundamentals: list[dict[str, object]],
        prices: list[dict[str, object]],
    ) -> CorpusRecord:
        ticker = str(snapshot["ticker"])
        as_of_time = _as_of_time(snapshot)
        latest_fundamentals = _latest_fundamentals(fundamentals, limit=3)
        latest_prices = _latest_prices(prices, limit=3)
        text = (
            f"Company: {snapshot['company_name']} ({ticker}).\n"
            f"Latest fundamental available as of: {snapshot.get('latest_fundamental_as_of_time') or 'unknown'}.\n"
            f"Observed price window: {snapshot.get('price_start') or 'unknown'} to {snapshot.get('price_end') or 'unknown'} "
            f"with {snapshot.get('price_observations', 0)} observations.\n"
            f"Recent fundamentals: {_format_fundamental_summary(latest_fundamentals)}.\n"
            f"Recent prices: {_format_price_summary(latest_prices)}.\n"
            "This record is generated from public SEC fundamentals and public historical prices for finance-domain CPT."
        )
        return CorpusRecord(
            corpus_id=f"public-cpt-{ticker}",
            task_type="cpt_text",
            mixture_name=mixture_name,
            entity_id=str(snapshot["entity_id"]),
            ticker=ticker,
            as_of_time=as_of_time,
            text=text,
            source_ids=[str(source_id) for source_id in snapshot.get("source_ids", [])],
            evidence_ids=[],
            quality_score=1.0,
        )

    def _snapshot_to_sft(
        self,
        mixture_name: str,
        snapshot: dict[str, object],
        fundamentals: list[dict[str, object]],
        prices: list[dict[str, object]],
    ) -> CorpusRecord:
        ticker = str(snapshot["ticker"])
        as_of_time = _as_of_time(snapshot)
        latest_fundamentals = _latest_fundamentals(fundamentals, limit=3)
        latest_prices = _latest_prices(prices, limit=2)
        evidence_ids = _evidence_ids(latest_fundamentals, latest_prices)
        answer = {
            "investment_view": "neutral",
            "confidence": 0.55 if evidence_ids else 0.0,
            "thesis": (
                f"{ticker} has usable public fundamentals and price coverage, but this MVP example should stay neutral "
                "because it does not yet include full filings, valuation comps, management commentary, or forward labels."
            ),
            "business_quality": _format_fundamental_summary(latest_fundamentals),
            "price_context": _format_price_summary(latest_prices),
            "risks": ["Evidence is limited to public company facts and historical prices."],
            "evidence_ids": evidence_ids,
        }
        text = (
            f"Question: As of {as_of_time}, produce an evidence-grounded long-term investment thesis for {ticker}. "
            "Use only the cited SEC fundamentals and historical price evidence.\n"
            f"Answer: {json.dumps(answer, sort_keys=True)}\n"
            f"Evidence: {', '.join(evidence_ids)}"
        )
        return CorpusRecord(
            corpus_id=f"public-sft-{ticker}",
            task_type="sft_instruction",
            mixture_name=mixture_name,
            entity_id=str(snapshot["entity_id"]),
            ticker=ticker,
            as_of_time=as_of_time,
            text=text,
            source_ids=evidence_ids,
            evidence_ids=evidence_ids,
            quality_score=1.0 if evidence_ids else 0.0,
        )

    def _read_optional(self, path: str) -> list[dict[str, object]]:
        try:
            return self.store.read_jsonl(path)
        except ArtifactNotFoundError:
            return []


def _as_of_time(snapshot: dict[str, object]) -> str:
    latest = snapshot.get("latest_fundamental_as_of_time")
    if latest:
        return str(latest)
    price_end = snapshot.get("price_end")
    if price_end:
        return f"{price_end}T00:00:00Z"
    raise ReadinessError(f"snapshot {snapshot.get('snapshot_id', '')} has no training as_of_time")


def _group_by_ticker(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        ticker = str(row.get("ticker", ""))
        if ticker:
            grouped.setdefault(ticker, []).append(row)
    return grouped


def _latest_fundamentals(rows: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    return sorted(rows, key=lambda row: (str(row.get("as_of_time", "")), str(row.get("period_end", ""))), reverse=True)[:limit]


def _latest_prices(rows: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    return sorted(rows, key=lambda row: str(row.get("date", "")), reverse=True)[:limit]


def _format_fundamental_summary(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "no normalized SEC fundamentals available"
    return "; ".join(
        f"{row.get('concept')}={row.get('value')} {row.get('unit')} period_end={row.get('period_end')} as_of={row.get('as_of_time')}"
        for row in rows
    )


def _format_price_summary(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "no normalized price observations available"
    ordered = list(reversed(rows))
    first = ordered[0]
    last = ordered[-1]
    return f"{first.get('date')} adjusted_close={first.get('adjusted_close')} to {last.get('date')} adjusted_close={last.get('adjusted_close')}"


def _evidence_ids(fundamentals: list[dict[str, object]], prices: list[dict[str, object]]) -> list[str]:
    ids = [f"fundamental-{row['fundamental_id']}" for row in fundamentals if row.get("fundamental_id")]
    ids.extend(f"price-{row['price_id']}" for row in prices if row.get("price_id"))
    return ids
