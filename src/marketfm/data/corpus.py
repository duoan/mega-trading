"""Build model-facing corpora from normalized financial records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime

from marketfm.core.config import DataMixtureConfig
from marketfm.core.schemas import CorpusRecord, Manifest
from marketfm.core.store import ArtifactPaths, LocalObjectStore


class LeakageError(ValueError):
    """Raised when future evidence enters a model-visible artifact."""


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
        documents = self.store.read_jsonl("silver/documents/fixture.jsonl")
        qa_examples = self.store.read_jsonl("silver/qa/fixture.jsonl")
        preference_pairs = self.store.read_jsonl("silver/preference/fixture.jsonl")
        evidence = {row["evidence_id"]: row for row in self.store.read_jsonl("silver/evidence/fixture.jsonl")}

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
