"""Order-flow corpus and token-shard builder."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from mega_trading.config import BuildConfig
from mega_trading.core.schemas import Manifest
from mega_trading.core.store import LocalObjectStore
from mega_trading.tokenizer import BOS_TOKEN, EOS_TOKEN, MarketEventTokenizer

STREAM_CONTRACT = "paper-order-flow-token-v1"


@dataclass(frozen=True)
class BuildResult:
    event_path: str
    shard_path: str
    profile_path: str
    manifest_path: str
    tokenizer_path: str
    numpy_tokens_path: str
    numpy_ticker_ids_path: str
    numpy_metadata_path: str


class EventBuilder:
    """Build paper-style event token streams from normalized order-flow rows."""

    def __init__(self, store: LocalObjectStore, config: BuildConfig | None = None) -> None:
        self.store = store
        self.config = config or BuildConfig()

    def build(self) -> BuildResult:
        events_by_ticker = _events_by_ticker(self.store, self.config.source)
        if self.config.max_tickers is not None:
            selected = sorted(events_by_ticker)[: self.config.max_tickers]
            events_by_ticker = {ticker: events_by_ticker[ticker] for ticker in selected}
        events_by_ticker = {
            ticker: events
            for ticker, events in sorted(events_by_ticker.items())
            if len(events) >= self.config.min_events_per_ticker
        }
        if not events_by_ticker:
            raise ValueError("no tickers had enough order-flow events")

        event_path = f"stage=04_corpus/mixture={self.config.mixture_name}/events.jsonl"
        shard_path = f"stage=05_shards/mixture={self.config.mixture_name}/tokens.jsonl"
        profile_path = f"stage=05_shards/mixture={self.config.mixture_name}/tokens-profile.json"
        tokenizer_path = f"stage=05_shards/mixture={self.config.mixture_name}/tokenizer.json"
        manifest_path = f"manifests/build/{self.config.mixture_name}.json"

        event_rows = [event for ticker in sorted(events_by_ticker) for event in events_by_ticker[ticker]]
        self.store.write_jsonl(event_path, event_rows)
        tokenizer = MarketEventTokenizer.fit(
            event_rows,
            relative_price_bins=self.config.tokenizer_relative_price_bins,
            price_bins=self.config.tokenizer_price_bins,
            size_bins=self.config.tokenizer_size_bins,
            time_bins=self.config.tokenizer_time_bins,
            method=self.config.tokenizer_method,
            clip_quantile=self.config.tokenizer_clip_quantile,
        )
        self.store.write_json(tokenizer_path, tokenizer.to_dict())

        sequence_rows = list(_sequence_rows(events_by_ticker, tokenizer, self.config.block_size, self.config.stride))
        if not sequence_rows:
            raise ValueError("not enough events to build token sequences")
        self.store.write_jsonl(shard_path, sequence_rows)
        numpy_metadata = _write_numpy_dataset(self.store, self.config.mixture_name, sequence_rows)

        profile = _profile(event_rows, sequence_rows, tokenizer, self.config, tokenizer_path, numpy_metadata)
        self.store.write_json(profile_path, profile)
        self.store.write_manifest(
            manifest_path,
            Manifest(
                manifest_id=f"{self.config.mixture_name}-build",
                artifact_type="event-token-build",
                paths=[
                    event_path,
                    shard_path,
                    profile_path,
                    tokenizer_path,
                    str(numpy_metadata["tokens_path"]),
                    str(numpy_metadata["ticker_ids_path"]),
                    str(numpy_metadata["metadata_path"]),
                ],
                metadata=profile,
            ),
        )
        return BuildResult(
            event_path=event_path,
            shard_path=shard_path,
            profile_path=profile_path,
            manifest_path=manifest_path,
            tokenizer_path=tokenizer_path,
            numpy_tokens_path=str(numpy_metadata["tokens_path"]),
            numpy_ticker_ids_path=str(numpy_metadata["ticker_ids_path"]),
            numpy_metadata_path=str(numpy_metadata["metadata_path"]),
        )


def _events_by_ticker(store: LocalObjectStore, source: str) -> dict[str, list[dict[str, Any]]]:
    path = f"stage=02_normalized/family=order_flow/source={source}.jsonl"
    rows = store.read_jsonl(path)
    events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        event = {
            "event_id": str(row["event_id"]),
            "ticker": str(row["ticker"]).upper(),
            "timestamp": str(row["timestamp"]),
            "date": str(row["date"]),
            "action": str(row["action"]),
            "side": str(row["side"]),
            "midprice": float(row["midprice"]),
            "relative_price_bps": float(row["relative_price_bps"]),
            "price_depth_bps": float(row["price_depth_bps"]),
            "size": float(row["size"]),
            "interarrival_seconds": float(row["interarrival_seconds"]),
            "source_ids": list(row["source_ids"]),
        }
        if row.get("midprice_return_bps") is not None:
            event["midprice_return_bps"] = float(row["midprice_return_bps"])
        events[event["ticker"]].append(event)
    return {ticker: sorted(values, key=lambda event: str(event["timestamp"])) for ticker, values in events.items()}


def _sequence_rows(
    events_by_ticker: dict[str, list[dict[str, Any]]],
    tokenizer: MarketEventTokenizer,
    block_size: int,
    stride: int,
) -> Iterable[dict[str, Any]]:
    sequence_id = 0
    target_length = block_size + 1
    per_ticker_rows: list[list[dict[str, Any]]] = []
    for ticker, events in sorted(events_by_ticker.items()):
        tokens = [BOS_TOKEN]
        timestamps = ["bos"]
        for event in events:
            tokens.extend(tokenizer.encode_event(event))
            timestamps.extend([str(event["timestamp"])] * tokenizer.event_size)
        tokens.append(EOS_TOKEN)
        timestamps.append("eos")

        rows: list[dict[str, Any]] = []
        for start in range(0, max(len(tokens) - target_length + 1, 0), stride):
            chunk = tokens[start : start + target_length]
            if len(chunk) == target_length:
                rows.append(
                    {
                        "sequence_id": f"seq-{sequence_id:08d}",
                        "ticker": ticker,
                        "start_time": timestamps[start],
                        "end_time": timestamps[start + target_length - 1],
                        "tokens": chunk,
                    }
                )
                sequence_id += 1
        per_ticker_rows.append(rows)

    for rows in _round_robin(per_ticker_rows):
        yield rows


def _profile(
    events: list[dict[str, Any]],
    sequences: list[dict[str, Any]],
    tokenizer: MarketEventTokenizer,
    config: BuildConfig,
    tokenizer_path: str,
    numpy_metadata: dict[str, Any],
) -> dict[str, Any]:
    ticker_counts = Counter(str(event["ticker"]) for event in events)
    sequence_counts = Counter(str(row["ticker"]) for row in sequences)
    return {
        "stream_contract": STREAM_CONTRACT,
        "feature_contract": "paper-order-flow-v1",
        "feature_order": ["action", "side", "relative_price", "price_depth", "size", "time"],
        "mixture": config.mixture_name,
        "source": config.source,
        "event_count": len(events),
        "sequence_count": len(sequences),
        "ticker_counts": dict(sorted(ticker_counts.items())),
        "sequence_counts": dict(sorted(sequence_counts.items())),
        "block_size": config.block_size,
        "stride": config.stride,
        "event_size": tokenizer.event_size,
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_path": tokenizer_path,
        "tokenizer_method": tokenizer.binning_method,
        "tokenizer_clip_quantile": tokenizer.clip_quantile,
        "tokenizer_bucket_counts": tokenizer.bucket_counts,
        "numpy_dataset": numpy_metadata,
    }


def _write_numpy_dataset(store: LocalObjectStore, mixture_name: str, sequences: list[dict[str, Any]]) -> dict[str, Any]:
    tokens_path = f"stage=05_shards/mixture={mixture_name}/tokens.npy"
    ticker_ids_path = f"stage=05_shards/mixture={mixture_name}/ticker_ids.npy"
    metadata_path = f"stage=05_shards/mixture={mixture_name}/tokens-numpy.json"
    tickers = sorted({str(row["ticker"]) for row in sequences})
    ticker_to_id = {ticker: index for index, ticker in enumerate(tickers)}
    token_array = np.asarray([row["tokens"] for row in sequences], dtype=np.int64)
    ticker_id_array = np.asarray([ticker_to_id[str(row["ticker"])] for row in sequences], dtype=np.int32)
    _write_npy(store, tokens_path, token_array)
    _write_npy(store, ticker_ids_path, ticker_id_array)
    metadata: dict[str, Any] = {
        "format": "mega-trading-numpy-token-v1",
        "tokens_path": tokens_path,
        "ticker_ids_path": ticker_ids_path,
        "metadata_path": metadata_path,
        "sequence_count": int(token_array.shape[0]),
        "sequence_length": int(token_array.shape[1]),
        "tokens_dtype": str(token_array.dtype),
        "ticker_ids_dtype": str(ticker_id_array.dtype),
        "ticker_to_id": ticker_to_id,
        "id_to_ticker": {str(index): ticker for ticker, index in ticker_to_id.items()},
    }
    store.write_json(metadata_path, metadata)
    return metadata


def _write_npy(store: LocalObjectStore, path: str, value: np.ndarray) -> None:
    target = _artifact_target(store, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, value)


def _artifact_target(store: LocalObjectStore, path: str) -> Path:
    target = Path(path)
    if target.is_absolute() or ".." in target.parts:
        raise ValueError(f"artifact path must be relative and safe: {path}")
    return store.root / target


def _round_robin(groups: list[list[dict[str, Any]]]) -> Iterable[dict[str, Any]]:
    max_len = max((len(group) for group in groups), default=0)
    for index in range(max_len):
        for group in groups:
            if index < len(group):
                yield group[index]
