"""Feature shard building for TradingFoundationModel training."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Iterable

from mega_trading.core.schemas import Manifest
from mega_trading.core.store import ArtifactPaths, LocalObjectStore


class MalformedSampleError(ValueError):
    """Raised when a model sample cannot be packed into a stream shard."""


@dataclass(frozen=True)
class StreamShardBuildResult:
    shard_path: str
    manifest_path: str
    num_samples: int


class StreamShardBuilder:
    """Build compact numeric shards for TradingFoundationModel training."""

    def __init__(self, store: LocalObjectStore, num_workers: int = 0) -> None:
        self.store = store
        self.num_workers = num_workers
        self.paths = ArtifactPaths()

    def build(self, mixture_name: str) -> StreamShardBuildResult:
        sample_path = self.paths.corpus(mixture_name, "samples")
        sample_paths = _sample_paths(self.store, mixture_name, sample_path)
        sample_count = sum(_jsonl_count(self.store, path) for path in sample_paths)
        workers = _worker_count(self.num_workers, len(sample_paths))
        if workers > 1:
            print(f"stream shard build using {workers} workers for {sample_count} samples across {len(sample_paths)} partitions")
        shard_path = self.paths.shard(mixture_name, "samples")
        shard_count = self.store.write_jsonl_iter(shard_path, _shard_rows(self.store.root, sample_paths, workers))

        manifest = Manifest(
            manifest_id=f"{mixture_name}-samples-shards",
            artifact_type="shards",
            paths=[shard_path],
            metadata={
                "artifact": "multi_stream_samples",
                "sample_path": sample_path,
                "sample_partitions": str(len(sample_paths)),
                "num_samples": str(shard_count),
                "workers": str(workers),
            },
        )
        manifest_path = self.paths.manifest("shards", f"{mixture_name}-samples-shards")
        self.store.write_manifest(manifest_path, manifest)
        return StreamShardBuildResult(shard_path=shard_path, manifest_path=manifest_path, num_samples=shard_count)


def _sample_paths(store: LocalObjectStore, mixture_name: str, fallback_sample_path: str) -> list[str]:
    partition_root = store.root / f"stage=04_corpus/mixture={mixture_name}/partitions"
    if partition_root.exists():
        paths = sorted(partition_root.glob("ticker=*/samples.jsonl"))
        if paths:
            return [str(path.relative_to(store.root)) for path in paths]
    return [fallback_sample_path]


def _shard_rows(store_root: object, sample_paths: list[str], workers: int) -> Iterable[dict[str, object]]:
    if workers <= 1:
        root = os.fspath(store_root)
        for sample_path in sample_paths:
            yield from _stream_partition((root, sample_path))
        return
    root = os.fspath(store_root)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for rows in executor.map(_stream_partition, [(root, sample_path) for sample_path in sample_paths], chunksize=1):
            yield from rows


def _stream_partition(task: tuple[str, str]) -> list[dict[str, object]]:
    root, sample_path = task
    rows: list[dict[str, object]] = []
    with open(os.path.join(root, sample_path), encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(_stream_row(json.loads(line)))
    return rows


def _stream_row(sample: dict[str, object]) -> dict[str, object]:
    market_data = _list_of_dicts(sample, "market_data_window")
    news = _list_of_dicts(sample, "news_window")
    sec_filings = _list_of_dicts(sample, "sec_filing_window")
    earnings = _list_of_dicts(sample, "earnings_window")
    macro = _list_of_dicts(sample, "macro_window")
    labels = sample.get("labels")
    if not isinstance(labels, dict):
        raise MalformedSampleError("sample row missing labels")
    return {
        "sample_id": _required_sample_string(sample, "sample_id"),
        "ticker": _required_sample_string(sample, "ticker"),
        "as_of_time": _required_sample_string(sample, "as_of_time"),
        "market_returns": _market_returns(market_data),
        "market_levels": _market_levels(market_data),
        "news_embeddings": _flatten_numeric(news, ("embedding", "sentiment", "importance_score")),
        "sec_filing_features": _flatten_numeric(sec_filings, ("embedding", "sentiment", "importance_score", "value")),
        "earnings_features": _flatten_numeric(earnings, ("embedding", "surprise", "eps_actual", "eps_estimate", "revenue_actual", "revenue_estimate")),
        "macro_features": _flatten_numeric(macro, ("regime_features", "vix", "interest_rate", "cpi", "yield_curve", "oil", "dxy")),
        "return_label": str(labels.get("forward_return_bucket", "")),
        "risk_label": str(labels.get("risk_bucket", "")),
        "forward_return": float(labels.get("forward_return", 0.0)),
        "source_ids": [str(source_id) for source_id in sample.get("source_ids", [])],
    }


def _list_of_dicts(row: dict[str, object], field: str) -> list[dict[str, object]]:
    value = row.get(field)
    if not isinstance(value, list):
        raise MalformedSampleError(f"sample row missing {field}")
    return [item for item in value if isinstance(item, dict)]


def _required_sample_string(row: dict[str, object], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise MalformedSampleError(f"sample row missing {field}")
    return value


def _market_returns(market_data: list[dict[str, object]]) -> list[float]:
    closes = _adjusted_closes(market_data)
    returns = [0.0]
    for index in range(1, len(closes)):
        previous = closes[index - 1]
        returns.append(0.0 if previous == 0 else _round_feature((closes[index] / previous) - 1.0))
    return returns


def _market_levels(market_data: list[dict[str, object]]) -> list[float]:
    closes = _adjusted_closes(market_data)
    if not closes or closes[0] == 0:
        return [0.0 for _ in closes]
    base = closes[0]
    return [_round_feature((close / base) - 1.0) for close in closes]


def _adjusted_closes(market_data: list[dict[str, object]]) -> list[float]:
    return [float(row["adjusted_close"]) for row in market_data if row.get("adjusted_close") is not None]


def _flatten_numeric(rows: list[dict[str, object]], fields: tuple[str, ...]) -> list[float]:
    values: list[float] = []
    for row in rows:
        for field in fields:
            values.extend(_numeric_values(row.get(field)))
    return values


def _numeric_values(value: object) -> list[float]:
    if value is None:
        return []
    if isinstance(value, list):
        return [number for item in value for number in _numeric_values(item)]
    if isinstance(value, (int, float)):
        return [_round_feature(float(value))]
    return []


def _round_feature(value: float) -> float:
    return round(value, 12)


def _jsonl_count(store: LocalObjectStore, path: str) -> int:
    return sum(1 for _ in store.iter_jsonl(path))


def _worker_count(requested_workers: int, task_count: int) -> int:
    if task_count <= 1:
        return 1
    if requested_workers == 0:
        requested_workers = os.cpu_count() or 1
    return max(1, min(requested_workers, task_count))
