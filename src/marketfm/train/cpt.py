"""Tiny CPT smoke trainer.

This is deliberately a tiny bigram language-model trainer. It validates the
training-system contracts before adding heavyweight ML dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter

from marketfm.core.hashing import stable_hash
from marketfm.core.schemas import Manifest
from marketfm.core.store import ArtifactPaths, LocalObjectStore


@dataclass(frozen=True)
class TrainConfig:
    run_id: str
    max_steps: int
    learning_rate: float = 0.2
    resume_from: str | None = None

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

    def content_hash(self) -> str:
        return stable_hash(
            {
                "run_id": self.run_id,
                "max_steps": self.max_steps,
                "learning_rate": self.learning_rate,
            }
        )


@dataclass(frozen=True)
class TrainResult:
    steps: int
    checkpoint_path: str
    manifest_path: str
    resume_used: bool


class CPTTrainer:
    def __init__(self, store: LocalObjectStore, config: TrainConfig) -> None:
        self.store = store
        self.config = config
        self.paths = ArtifactPaths(run_id=config.run_id)

    def train(self, shard_path: str) -> TrainResult:
        sequences = [row["token_ids"] for row in self.store.read_jsonl(shard_path)]
        pairs = _bigram_pairs(sequences)
        counts: dict[str, float] = {}
        start_step = 0
        resume_used = False

        if self.config.resume_from:
            checkpoint = self.store.read_json(self.config.resume_from)
            start_step = int(checkpoint["step"])
            counts = {key: float(value) for key, value in checkpoint["counts"].items()}
            resume_used = True

        metrics_path = self.paths.run("metrics")
        existing_metrics = self.store.read_jsonl(metrics_path) if resume_used else []
        metrics = list(existing_metrics)

        for step in range(start_step + 1, self.config.max_steps + 1):
            started = perf_counter()
            loss = self._train_step(pairs, counts)
            elapsed = max(perf_counter() - started, 1e-9)
            metrics.append(
                {
                    "step": step,
                    "stage": "cpt",
                    "loss": loss,
                    "tokens_per_second": len(pairs) / elapsed,
                    "examples_per_second": len(sequences) / elapsed,
                }
            )
            self.store.write_jsonl(metrics_path, metrics)

        checkpoint_path = self.paths.run("checkpoint.json")
        self.store.write_json(
            checkpoint_path,
            {
                "stage": "cpt",
                "step": self.config.max_steps,
                "config_hash": self.config.content_hash(),
                "shard_path": shard_path,
                "counts": counts,
            },
        )

        manifest = Manifest(
            manifest_id=f"{self.config.run_id}-cpt",
            artifact_type="training_run",
            paths=[metrics_path, checkpoint_path],
            metadata={
                "stage": "cpt",
                "steps": str(self.config.max_steps),
                "shard_path": shard_path,
                "config_hash": self.config.content_hash(),
                "resume_used": str(resume_used),
            },
        )
        manifest_path = self.paths.manifest("runs", f"{self.config.run_id}-cpt")
        self.store.write_manifest(manifest_path, manifest)

        return TrainResult(
            steps=self.config.max_steps,
            checkpoint_path=checkpoint_path,
            manifest_path=manifest_path,
            resume_used=resume_used,
        )

    def _train_step(self, pairs: list[tuple[int, int]], counts: dict[str, float]) -> float:
        loss = 0.0
        vocab_size = max([target for _, target in pairs], default=1) + 1
        for source, target in pairs:
            key = _pair_key(source, target)
            source_total = sum(value for pair_key, value in counts.items() if pair_key.startswith(f"{source}:"))
            pair_count = counts.get(key, 0.0)
            probability = (pair_count + 1.0) / (source_total + vocab_size)
            loss += -math.log(probability)
            counts[key] = pair_count + self.config.learning_rate
        return loss / max(1, len(pairs))


def _bigram_pairs(sequences: list[object]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for sequence in sequences:
        if not isinstance(sequence, list):
            continue
        token_ids = [int(token_id) for token_id in sequence]
        pairs.extend(zip(token_ids, token_ids[1:]))
    return pairs


def _pair_key(source: int, target: int) -> str:
    return f"{source}:{target}"
