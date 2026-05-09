"""Online adaptation interface for adapter/head updates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from mega_trading.core.schemas import Manifest
from mega_trading.core.store import ArtifactPaths, LocalObjectStore
from mega_trading.eval.replay_buffer import ReplayBufferConfig, select_replay_batch


@dataclass(frozen=True)
class OnlineAdaptationConfig:
    replay_buffer: ReplayBufferConfig = ReplayBufferConfig()
    learning_rate: float = 1e-4

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


@dataclass(frozen=True)
class OnlineAdaptationResult:
    adapter_path: str
    manifest_path: str
    samples: int


def run_online_adapter_update(
    store: LocalObjectStore,
    *,
    update_id: str,
    labeled_prediction_path: str,
    base_model_version: str,
    config: OnlineAdaptationConfig | None = None,
) -> OnlineAdaptationResult:
    config = config or OnlineAdaptationConfig()
    rows = [row for row in store.iter_jsonl(labeled_prediction_path) if row.get("label_status") == "ready"]
    batch = select_replay_batch(rows, config.replay_buffer)
    adapter_path = f"models/adapters/{update_id}.json"
    adapter_record = {
        "adapter_version": f"adapter-{update_id}",
        "base_model_version": base_model_version,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "stub",
        "frozen_components": [
            "market_data_encoder",
            "news_encoder",
            "sec_filing_encoder",
            "earnings_encoder",
            "macro_encoder",
            "fusion_backbone",
        ],
        "trainable_components": ["adapter", "task_head", "calibrator", "regime_embedding"],
        "sample_count": len(batch),
        "source_prediction_path": labeled_prediction_path,
        "learning_rate": config.learning_rate,
        "replay_buffer": asdict(config.replay_buffer),
    }
    store.write_json(adapter_path, adapter_record)
    manifest = Manifest(
        manifest_id=f"{update_id}-online-adapter-update",
        artifact_type="online_adapter_update",
        paths=[adapter_path],
        metadata={
            "update_id": update_id,
            "base_model_version": base_model_version,
            "labeled_prediction_path": labeled_prediction_path,
            "samples": str(len(batch)),
            "status": "stub",
        },
    )
    manifest_path = ArtifactPaths().manifest("models", f"{update_id}-online-adapter-update")
    store.write_manifest(manifest_path, manifest)
    return OnlineAdaptationResult(adapter_path=adapter_path, manifest_path=manifest_path, samples=len(batch))
