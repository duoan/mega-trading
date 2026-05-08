"""SFT smoke path for evidence-grounded investment reasoning."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from marketfm.schemas import Manifest, SchemaValidationError
from marketfm.store import ArtifactPaths, LocalObjectStore


@dataclass(frozen=True)
class ThesisOutput:
    ticker: str
    as_of_time: str
    investment_view: str
    confidence: float
    thesis: str
    evidence_ids: list[str]

    def __post_init__(self) -> None:
        if self.investment_view not in {"attractive", "neutral", "unattractive", "insufficient_evidence"}:
            raise SchemaValidationError(f"invalid investment_view: {self.investment_view}")
        if not 0.0 <= self.confidence <= 1.0:
            raise SchemaValidationError("confidence must be between 0 and 1")
        if not self.evidence_ids:
            raise SchemaValidationError("evidence_ids are required")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SFTExample:
    example_id: str
    prompt: str
    target: str
    evidence_ids: list[str]


@dataclass(frozen=True)
class SFTTrainConfig:
    run_id: str
    stage: str = "sft"


@dataclass(frozen=True)
class SFTTrainResult:
    examples: int
    manifest_path: str


class SFTExampleFormatter:
    def __init__(self, store: LocalObjectStore) -> None:
        self.store = store

    def load(self, corpus_path: str) -> list[SFTExample]:
        rows = self.store.read_jsonl(corpus_path)
        examples: list[SFTExample] = []
        for row in rows:
            text = str(row["text"])
            prompt, target = _split_question_answer(text)
            examples.append(
                SFTExample(
                    example_id=str(row["corpus_id"]),
                    prompt=prompt,
                    target=f"{target}\nEvidence: {', '.join(row['evidence_ids'])}",
                    evidence_ids=[str(evidence_id) for evidence_id in row["evidence_ids"]],
                )
            )
        return examples


class SFTTrainer:
    """A smoke trainer that validates SFT artifacts before real LoRA training."""

    def __init__(self, store: LocalObjectStore, config: SFTTrainConfig) -> None:
        self.store = store
        self.config = config
        self.paths = ArtifactPaths(run_id=config.run_id)

    def train(self, corpus_path: str) -> SFTTrainResult:
        examples = SFTExampleFormatter(self.store).load(corpus_path)
        citation_examples = sum(1 for example in examples if example.evidence_ids)
        schema_compliance = citation_examples / max(1, len(examples))

        metrics_path = self.paths.run("metrics")
        self.store.write_jsonl(
            metrics_path,
            [
                {
                    "stage": self.config.stage,
                    "examples": len(examples),
                    "schema_compliance": schema_compliance,
                    "citation_inclusion_rate": schema_compliance,
                }
            ],
        )

        examples_path = self.paths.run("sft_examples.jsonl")
        self.store.write_jsonl(examples_path, [asdict(example) for example in examples])

        manifest = Manifest(
            manifest_id=f"{self.config.run_id}-sft",
            artifact_type="training_run",
            paths=[metrics_path, examples_path],
            metadata={
                "stage": self.config.stage,
                "examples": str(len(examples)),
                "corpus_path": corpus_path,
                "schema_compliance": f"{schema_compliance:.6f}",
            },
        )
        manifest_path = self.paths.manifest("runs", f"{self.config.run_id}-sft")
        self.store.write_manifest(manifest_path, manifest)

        return SFTTrainResult(examples=len(examples), manifest_path=manifest_path)


def _split_question_answer(text: str) -> tuple[str, str]:
    lines = text.splitlines()
    question = next((line.removeprefix("Question: ").strip() for line in lines if line.startswith("Question: ")), "")
    answer = next((line.removeprefix("Answer: ").strip() for line in lines if line.startswith("Answer: ")), "")
    if not question or not answer:
        raise SchemaValidationError("SFT corpus text must include Question and Answer lines")
    return question, answer
