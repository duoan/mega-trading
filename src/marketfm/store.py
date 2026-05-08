"""Local object-store abstraction for deterministic demos."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from marketfm.schemas import Manifest


class ArtifactNotFoundError(FileNotFoundError):
    """Raised when an expected artifact path is missing."""


@dataclass(frozen=True)
class ArtifactPaths:
    run_id: str = "demo"

    def bronze(self, source: str, name: str) -> str:
        return f"bronze/{source}/{name}.jsonl"

    def silver(self, family: str, name: str) -> str:
        return f"silver/{family}/{name}.jsonl"

    def corpus(self, mixture: str, name: str) -> str:
        return f"corpus/{mixture}/{name}.jsonl"

    def manifest(self, family: str, manifest_id: str) -> str:
        return f"manifests/{family}/{manifest_id}.json"

    def run(self, name: str) -> str:
        return f"runs/{self.run_id}/{name}.jsonl"


class LocalObjectStore:
    """A filesystem-backed store with S3-like relative artifact paths."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _resolve(self, path: str) -> Path:
        if path.startswith("/") or ".." in Path(path).parts:
            raise ValueError(f"artifact path must be relative and safe: {path}")
        return self.root / path

    def write_json(self, path: str, value: dict[str, Any]) -> None:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    def read_json(self, path: str) -> dict[str, Any]:
        target = self._resolve(path)
        if not target.exists():
            raise ArtifactNotFoundError(path)
        return json.loads(target.read_text(encoding="utf-8"))

    def write_jsonl(self, path: str, rows: list[dict[str, Any]]) -> None:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        target.write_text(content, encoding="utf-8")

    def read_jsonl(self, path: str) -> list[dict[str, Any]]:
        target = self._resolve(path)
        if not target.exists():
            raise ArtifactNotFoundError(path)
        return [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line]

    def write_manifest(self, path: str, manifest: Manifest) -> None:
        self.write_json(path, manifest.to_dict())

    def read_manifest(self, path: str) -> Manifest:
        value = self.read_json(path)
        return Manifest(
            manifest_id=value["manifest_id"],
            artifact_type=value["artifact_type"],
            paths=list(value["paths"]),
            metadata=dict(value.get("metadata", {})),
        )
