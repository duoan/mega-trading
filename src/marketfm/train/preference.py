"""Preference-training smoke path for DPO-style alignment."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from marketfm.schemas import Manifest, SchemaValidationError
from marketfm.store import ArtifactPaths, LocalObjectStore


@dataclass(frozen=True)
class PreferencePair:
    pair_id: str
    prompt: str
    chosen: str
    rejected: str
    evidence_ids: list[str]

    def __post_init__(self) -> None:
        if not self.evidence_ids:
            raise SchemaValidationError("preference pair requires evidence_ids")


@dataclass(frozen=True)
class PreferenceTrainConfig:
    run_id: str
    stage: str = "preference"


@dataclass(frozen=True)
class PreferenceTrainResult:
    pairs: int
    manifest_path: str


class PreferencePairLoader:
    def __init__(self, store: LocalObjectStore) -> None:
        self.store = store

    def load(self, corpus_path: str) -> list[PreferencePair]:
        rows = self.store.read_jsonl(corpus_path)
        grouped: dict[str, dict[str, object]] = {}
        for row in rows:
            corpus_id = str(row["corpus_id"])
            pair_id = corpus_id.removeprefix("pref-").rsplit("-", maxsplit=1)[0]
            label = corpus_id.rsplit("-", maxsplit=1)[1]
            grouped.setdefault(
                pair_id,
                {
                    "prompt": _extract_prompt(str(row["text"])),
                    "evidence_ids": [str(evidence_id) for evidence_id in row["evidence_ids"]],
                },
            )[label] = _extract_response(str(row["text"]))

        pairs: list[PreferencePair] = []
        for pair_id, value in grouped.items():
            pairs.append(
                PreferencePair(
                    pair_id=pair_id,
                    prompt=str(value["prompt"]),
                    chosen=str(value["chosen"]),
                    rejected=str(value["rejected"]),
                    evidence_ids=list(value["evidence_ids"]),
                )
            )
        return pairs


class PreferenceTrainer:
    """Smoke trainer that records preference data quality metrics."""

    def __init__(self, store: LocalObjectStore, config: PreferenceTrainConfig) -> None:
        self.store = store
        self.config = config
        self.paths = ArtifactPaths(run_id=config.run_id)

    def train(self, corpus_path: str) -> PreferenceTrainResult:
        pairs = PreferencePairLoader(self.store).load(corpus_path)
        chosen_evidence_rate = sum(_contains_any(pair.chosen, pair.evidence_ids) for pair in pairs) / max(1, len(pairs))
        rejected_evidence_rate = sum(_contains_any(pair.rejected, pair.evidence_ids) for pair in pairs) / max(1, len(pairs))
        preference_margin = chosen_evidence_rate - rejected_evidence_rate

        metrics_path = self.paths.run("metrics")
        self.store.write_jsonl(
            metrics_path,
            [
                {
                    "stage": self.config.stage,
                    "preference_pairs": len(pairs),
                    "chosen_evidence_rate": chosen_evidence_rate,
                    "rejected_evidence_rate": rejected_evidence_rate,
                    "preference_margin": preference_margin,
                }
            ],
        )

        pairs_path = self.paths.run("preference_pairs.jsonl")
        self.store.write_jsonl(pairs_path, [asdict(pair) for pair in pairs])

        manifest = Manifest(
            manifest_id=f"{self.config.run_id}-preference",
            artifact_type="training_run",
            paths=[metrics_path, pairs_path],
            metadata={
                "stage": self.config.stage,
                "pairs": str(len(pairs)),
                "corpus_path": corpus_path,
                "preference_margin": f"{preference_margin:.6f}",
            },
        )
        manifest_path = self.paths.manifest("runs", f"{self.config.run_id}-preference")
        self.store.write_manifest(manifest_path, manifest)

        return PreferenceTrainResult(pairs=len(pairs), manifest_path=manifest_path)


def _extract_prompt(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("Prompt: "):
            return line.removeprefix("Prompt: ").strip()
    raise SchemaValidationError("preference text missing Prompt line")


def _extract_response(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("Response: "):
            return line.removeprefix("Response: ").strip()
    raise SchemaValidationError("preference text missing Response line")


def _contains_any(text: str, evidence_ids: list[str]) -> bool:
    return any(evidence_id in text for evidence_id in evidence_ids)
