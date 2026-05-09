"""Replay-buffer sampling for online adaptation."""

from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class ReplayBufferConfig:
    max_samples: int = 128
    recent_fraction: float = 0.7
    same_regime_fraction: float = 0.2
    random_fraction: float = 0.1
    seed: int = 7
    regime_field: str = "actual_risk_bucket"

    def __post_init__(self) -> None:
        if self.max_samples <= 0:
            raise ValueError("max_samples must be positive")
        total = self.recent_fraction + self.same_regime_fraction + self.random_fraction
        if abs(total - 1.0) > 1e-9:
            raise ValueError("replay buffer fractions must sum to 1.0")


def select_replay_batch(
    rows: list[dict[str, object]],
    config: ReplayBufferConfig | None = None,
    target_regime: str | None = None,
) -> list[dict[str, object]]:
    config = config or ReplayBufferConfig()
    if not rows:
        return []
    target_regime = target_regime or str(rows[-1].get(config.regime_field, ""))
    budget = min(config.max_samples, len(rows))
    recent_count = min(budget, int(budget * config.recent_fraction))
    same_regime_count = min(budget - recent_count, int(budget * config.same_regime_fraction))
    random_count = budget - recent_count - same_regime_count

    selected = list(rows[-recent_count:]) if recent_count else []
    selected_ids = {_row_id(row) for row in selected}
    same_regime = [
        row for row in rows if _row_id(row) not in selected_ids and str(row.get(config.regime_field, "")) == target_regime
    ]
    selected.extend(same_regime[:same_regime_count])
    selected_ids = {_row_id(row) for row in selected}

    remaining = [row for row in rows if _row_id(row) not in selected_ids]
    rng = random.Random(config.seed)
    rng.shuffle(remaining)
    selected.extend(remaining[:random_count])
    return selected[:budget]


def _row_id(row: dict[str, object]) -> str:
    return str(row.get("prediction_id") or row.get("sample_id") or id(row))
