"""Tiny fusion-model smoke trainer for multi-stream market samples."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from marketfm.core.hashing import stable_hash
from marketfm.core.schemas import Manifest
from marketfm.core.store import ArtifactPaths, LocalObjectStore


RETURN_LABELS = ("underperform", "neutral", "outperform")
RISK_LABELS = ("low", "medium", "high")


@dataclass(frozen=True)
class FusionTrainConfig:
    run_id: str
    max_steps: int
    learning_rate: float = 0.1
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
class FusionTrainResult:
    steps: int
    checkpoint_path: str
    manifest_path: str
    resume_used: bool


class TinyFusionTrainer:
    """A dependency-free classifier that validates the fusion training contract."""

    def __init__(self, store: LocalObjectStore, config: FusionTrainConfig) -> None:
        self.store = store
        self.config = config
        self.paths = ArtifactPaths(run_id=config.run_id)

    def train(self, shard_path: str) -> FusionTrainResult:
        rows = self.store.read_jsonl(shard_path)
        weights = _zero_weights()
        start_step = 0
        resume_used = False
        if self.config.resume_from:
            checkpoint = self.store.read_json(self.config.resume_from)
            weights = {
                "return": {label: [float(value) for value in values] for label, values in checkpoint["weights"]["return"].items()},
                "risk": {label: [float(value) for value in values] for label, values in checkpoint["weights"]["risk"].items()},
            }
            start_step = int(checkpoint["step"])
            resume_used = True

        metrics_path = self.paths.run("metrics")
        metrics = self.store.read_jsonl(metrics_path) if resume_used else []
        for step in range(start_step + 1, self.config.max_steps + 1):
            started = perf_counter()
            return_correct, risk_correct, loss = self._train_step(rows, weights)
            elapsed = max(perf_counter() - started, 1e-9)
            metrics.append(
                {
                    "step": step,
                    "stage": "fusion",
                    "loss": loss,
                    "return_accuracy": return_correct / max(1, len(rows)),
                    "risk_accuracy": risk_correct / max(1, len(rows)),
                    "examples_per_second": len(rows) / elapsed,
                }
            )
            self.store.write_jsonl(metrics_path, metrics)

        checkpoint_path = self.paths.run("checkpoint.json")
        self.store.write_json(
            checkpoint_path,
            {
                "stage": "fusion",
                "step": self.config.max_steps,
                "config_hash": self.config.content_hash(),
                "shard_path": shard_path,
                "weights": weights,
            },
        )
        manifest = Manifest(
            manifest_id=f"{self.config.run_id}-fusion",
            artifact_type="training_run",
            paths=[metrics_path, checkpoint_path],
            metadata={
                "stage": "fusion",
                "steps": str(self.config.max_steps),
                "shard_path": shard_path,
                "stream_contract": "price_fundamental_text",
                "config_hash": self.config.content_hash(),
                "resume_used": str(resume_used),
            },
        )
        manifest_path = self.paths.manifest("runs", f"{self.config.run_id}-fusion")
        self.store.write_manifest(manifest_path, manifest)
        return FusionTrainResult(
            steps=self.config.max_steps,
            checkpoint_path=checkpoint_path,
            manifest_path=manifest_path,
            resume_used=resume_used,
        )

    def _train_step(self, rows: list[dict[str, object]], weights: dict[str, dict[str, list[float]]]) -> tuple[int, int, float]:
        return_correct = 0
        risk_correct = 0
        mistakes = 0
        for row in rows:
            features = _features(row)
            return_label = str(row["return_label"])
            risk_label = str(row["risk_label"])
            return_prediction = _predict(features, weights["return"], RETURN_LABELS)
            risk_prediction = _predict(features, weights["risk"], RISK_LABELS)
            if return_prediction == return_label:
                return_correct += 1
            else:
                mistakes += 1
                _update(weights["return"], features, return_label, return_prediction, self.config.learning_rate)
            if risk_prediction == risk_label:
                risk_correct += 1
            else:
                mistakes += 1
                _update(weights["risk"], features, risk_label, risk_prediction, self.config.learning_rate)
        loss = mistakes / max(1, len(rows) * 2)
        return return_correct, risk_correct, loss


def _zero_weights() -> dict[str, dict[str, list[float]]]:
    return {
        "return": {label: [0.0] * 6 for label in RETURN_LABELS},
        "risk": {label: [0.0] * 6 for label in RISK_LABELS},
    }


def _features(row: dict[str, object]) -> list[float]:
    price_returns = _float_list(row.get("price_returns", []))
    price_levels = _float_list(row.get("price_levels", []))
    fundamentals = _float_list(row.get("fundamental_values", []))
    evidence_tokens = row.get("evidence_token_ids", [])
    evidence_count = len(evidence_tokens) if isinstance(evidence_tokens, list) else 0
    return [
        1.0,
        price_levels[-1] if price_levels else 0.0,
        sum(price_returns) / max(1, len(price_returns)),
        _max_drawdown(price_levels),
        sum(fundamentals) / max(1.0, len(fundamentals)) / 1_000_000_000.0,
        evidence_count / 100.0,
    ]


def _predict(features: list[float], weights: dict[str, list[float]], labels: tuple[str, ...]) -> str:
    return max(labels, key=lambda label: _dot(features, weights[label]))


def _update(
    weights: dict[str, list[float]],
    features: list[float],
    target: str,
    prediction: str,
    learning_rate: float,
) -> None:
    if target not in weights:
        return
    for index, value in enumerate(features):
        weights[target][index] += learning_rate * value
        weights[prediction][index] -= learning_rate * value


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _float_list(value: object) -> list[float]:
    if not isinstance(value, list):
        return []
    return [float(item) for item in value]


def _max_drawdown(levels: list[float]) -> float:
    if not levels:
        return 0.0
    peak = levels[0]
    drawdown = 0.0
    for level in levels:
        peak = max(peak, level)
        drawdown = min(drawdown, level - peak)
    return drawdown
