"""Model registry records for replayable platform runs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from mega_trading.core.schemas import ModelVersionRecord
from mega_trading.core.store import ArtifactPaths, LocalObjectStore


def model_version_id_for(run_id: str, config_hash: str) -> str:
    return f"{run_id}-{config_hash[:12]}"


class ModelRegistry:
    def __init__(self, store: LocalObjectStore) -> None:
        self.store = store
        self.paths = ArtifactPaths()

    def register_training_run(
        self,
        *,
        run_id: str,
        checkpoint_path: str,
        manifest_path: str,
        metrics_path: str,
        config_hash: str,
        feature_version: str,
        label_version: str,
        data_snapshot_version: str,
        metadata: dict[str, Any] | None = None,
    ) -> ModelVersionRecord:
        model_version_id = model_version_id_for(run_id, config_hash)
        record = ModelVersionRecord(
            model_version_id=model_version_id,
            run_id=run_id,
            checkpoint_path=checkpoint_path,
            manifest_path=manifest_path,
            metrics_path=metrics_path,
            base_model_version=f"base-{model_version_id}",
            adapter_version="adapter-none",
            head_version=f"head-{model_version_id}",
            feature_version=feature_version,
            label_version=label_version,
            data_snapshot_version=data_snapshot_version,
            config_hash=config_hash,
            created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            metadata=metadata or {},
        )
        self.store.write_json(self.paths.model(model_version_id), record.to_dict())
        return record

    def read_model_version(self, model_version_id: str) -> ModelVersionRecord:
        value = self.store.read_json(self.paths.model(model_version_id))
        return ModelVersionRecord(
            model_version_id=value["model_version_id"],
            run_id=value["run_id"],
            checkpoint_path=value["checkpoint_path"],
            manifest_path=value["manifest_path"],
            metrics_path=value["metrics_path"],
            base_model_version=value["base_model_version"],
            adapter_version=value["adapter_version"],
            head_version=value["head_version"],
            feature_version=value["feature_version"],
            label_version=value["label_version"],
            data_snapshot_version=value["data_snapshot_version"],
            config_hash=value["config_hash"],
            created_at=value["created_at"],
            metadata=dict(value.get("metadata", {})),
        )
