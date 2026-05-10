"""Order-flow corpus and token-shard builder."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np

from mega_trading.config import BuildConfig
from mega_trading.core.schemas import Manifest
from mega_trading.core.store import LocalObjectStore
from mega_trading.tokenizer import BOS_TOKEN, EOS_TOKEN, MarketEventTokenizer

STREAM_CONTRACT = "paper-order-flow-token-v1"


@dataclass(frozen=True)
class BuildResult:
    profile_path: str
    manifest_path: str
    tokenizer_path: str
    numpy_tokens_path: str
    numpy_ticker_ids_path: str
    numpy_metadata_path: str


class EventBuilder:
    """Build paper-style event token streams from normalized order-flow rows."""

    def __init__(
        self,
        store: LocalObjectStore,
        config: BuildConfig | None = None,
    ) -> None:
        self.store = store
        self.config = config or BuildConfig()

    def build_events(self, events_by_ticker: dict[str, list[dict[str, Any]]]) -> BuildResult:
        started_at = perf_counter()
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

        profile_path = f"datasets/mixture={self.config.mixture_name}/tokens-profile.json"
        tokenizer_path = f"datasets/mixture={self.config.mixture_name}/tokenizer.json"
        manifest_path = f"manifests/build/{self.config.mixture_name}.json"

        event_count = sum(len(events) for events in events_by_ticker.values())
        print(
            "event build: "
            f"fitting tokenizer on {event_count} events across {len(events_by_ticker)} tickers"
        )
        fit_started_at = perf_counter()
        tokenizer = MarketEventTokenizer.fit(
            _iter_events(events_by_ticker),
            relative_price_bins=self.config.tokenizer_relative_price_bins,
            price_bins=self.config.tokenizer_price_bins,
            size_bins=self.config.tokenizer_size_bins,
            time_bins=self.config.tokenizer_time_bins,
            method=self.config.tokenizer_method,
            clip_quantile=self.config.tokenizer_clip_quantile,
        )
        self.store.write_json(tokenizer_path, tokenizer.to_dict())
        print(f"event build: fitted tokenizer in {perf_counter() - fit_started_at:.1f}s")

        sequence_started_at = perf_counter()
        sequence_rows = list(_sequence_rows(events_by_ticker, tokenizer, self.config.block_size, self.config.stride))
        if not sequence_rows:
            raise ValueError("not enough events to build token sequences")
        print(
            "event build: "
            f"built {len(sequence_rows)} token sequences in {perf_counter() - sequence_started_at:.1f}s"
        )
        write_started_at = perf_counter()
        split_metadata = _split_metadata(sequence_rows, self.config.validation_fraction, self.config.backtest_fraction)
        numpy_metadata = _write_numpy_dataset(
            self.store,
            self.config.mixture_name,
            sequence_rows,
            partition_rows=self.config.numpy_partition_rows,
            split_metadata=split_metadata,
        )
        print(
            "event build: "
            f"wrote NumPy shards in {perf_counter() - write_started_at:.1f}s; "
            f"total build time={perf_counter() - started_at:.1f}s"
        )

        profile = _profile(events_by_ticker, sequence_rows, tokenizer, self.config, tokenizer_path, numpy_metadata)
        self.store.write_json(profile_path, profile)
        manifest_paths = [
            profile_path,
            tokenizer_path,
            str(numpy_metadata["tokens_path"]),
            str(numpy_metadata["ticker_ids_path"]),
            str(numpy_metadata["metadata_path"]),
        ]
        self.store.write_manifest(
            manifest_path,
            Manifest(
                manifest_id=f"{self.config.mixture_name}-build",
                artifact_type="event-token-build",
                paths=manifest_paths,
                metadata=profile,
            ),
        )
        return BuildResult(
            profile_path=profile_path,
            manifest_path=manifest_path,
            tokenizer_path=tokenizer_path,
            numpy_tokens_path=str(numpy_metadata["tokens_path"]),
            numpy_ticker_ids_path=str(numpy_metadata["ticker_ids_path"]),
            numpy_metadata_path=str(numpy_metadata["metadata_path"]),
        )


def events_by_ticker_from_rows(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
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


def events_by_ticker_from_records(records: Iterable[Any], *, assume_sorted: bool = False) -> dict[str, list[dict[str, Any]]]:
    events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        event = {
            "event_id": str(record.event_id),
            "ticker": str(record.ticker).upper(),
            "timestamp": str(record.timestamp),
            "date": str(record.date),
            "action": str(record.action),
            "side": str(record.side),
            "midprice": float(record.midprice),
            "relative_price_bps": float(record.relative_price_bps),
            "price_depth_bps": float(record.price_depth_bps),
            "size": float(record.size),
            "interarrival_seconds": float(record.interarrival_seconds),
            "source_ids": list(record.source_ids),
        }
        if getattr(record, "midprice_return_bps", None) is not None:
            event["midprice_return_bps"] = float(record.midprice_return_bps)
        events[event["ticker"]].append(event)
    if assume_sorted:
        return dict(sorted(events.items()))
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
    events_by_ticker: dict[str, list[dict[str, Any]]],
    sequences: list[dict[str, Any]],
    tokenizer: MarketEventTokenizer,
    config: BuildConfig,
    tokenizer_path: str,
    numpy_metadata: dict[str, Any],
) -> dict[str, Any]:
    ticker_counts = Counter({ticker: len(events) for ticker, events in events_by_ticker.items()})
    sequence_counts = Counter(str(row["ticker"]) for row in sequences)
    return {
        "stream_contract": STREAM_CONTRACT,
        "feature_contract": "paper-order-flow-v1",
        "feature_order": ["action", "side", "relative_price", "price_depth", "size", "time"],
        "mixture": config.mixture_name,
        "source": config.source,
        "event_count": sum(ticker_counts.values()),
        "sequence_count": len(sequences),
        "ticker_counts": dict(sorted(ticker_counts.items())),
        "sequence_counts": dict(sorted(sequence_counts.items())),
        "block_size": config.block_size,
        "stride": config.stride,
        "validation_fraction": config.validation_fraction,
        "backtest_fraction": config.backtest_fraction,
        "event_size": tokenizer.event_size,
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_path": tokenizer_path,
        "tokenizer_method": tokenizer.binning_method,
        "tokenizer_clip_quantile": tokenizer.clip_quantile,
        "tokenizer_bucket_counts": tokenizer.bucket_counts,
        "numpy_dataset": numpy_metadata,
    }


def _iter_events(events_by_ticker: dict[str, list[dict[str, Any]]]) -> Iterable[dict[str, Any]]:
    for ticker in sorted(events_by_ticker):
        yield from events_by_ticker[ticker]


def _write_numpy_dataset(
    store: LocalObjectStore,
    mixture_name: str,
    sequences: list[dict[str, Any]],
    partition_rows: int | None = None,
    split_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tokens_path = f"datasets/mixture={mixture_name}/tokens.npy"
    ticker_ids_path = f"datasets/mixture={mixture_name}/ticker_ids.npy"
    metadata_path = f"datasets/mixture={mixture_name}/tokens-numpy.json"
    tickers = sorted({str(row["ticker"]) for row in sequences})
    ticker_to_id = {ticker: index for index, ticker in enumerate(tickers)}
    token_array = np.asarray([row["tokens"] for row in sequences], dtype=np.int64)
    ticker_id_array = np.asarray([ticker_to_id[str(row["ticker"])] for row in sequences], dtype=np.int32)
    if partition_rows is not None and len(sequences) > partition_rows:
        return _write_partitioned_numpy_dataset(
            store,
            mixture_name,
            token_array,
            ticker_id_array,
            ticker_to_id,
            metadata_path,
            partition_rows,
            split_metadata,
        )
    store.delete_tree_if_exists(f"datasets/mixture={mixture_name}/numpy")
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
        "partitioned": False,
        "splits": split_metadata or {},
        "ticker_to_id": ticker_to_id,
        "id_to_ticker": {str(index): ticker for ticker, index in ticker_to_id.items()},
    }
    store.write_json(metadata_path, metadata)
    return metadata


def _write_partitioned_numpy_dataset(
    store: LocalObjectStore,
    mixture_name: str,
    token_array: np.ndarray,
    ticker_id_array: np.ndarray,
    ticker_to_id: dict[str, int],
    metadata_path: str,
    partition_rows: int,
    split_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    root_path = f"datasets/mixture={mixture_name}/numpy"
    store.delete_if_exists(f"datasets/mixture={mixture_name}/tokens.npy")
    store.delete_if_exists(f"datasets/mixture={mixture_name}/ticker_ids.npy")
    store.delete_tree_if_exists(root_path)
    partitions: list[dict[str, Any]] = []
    for partition_index, start in enumerate(range(0, int(token_array.shape[0]), partition_rows)):
        stop = min(start + partition_rows, int(token_array.shape[0]))
        partition_id = f"part-{partition_index:05d}"
        tokens_path = f"{root_path}/{partition_id}/tokens.npy"
        ticker_ids_path = f"{root_path}/{partition_id}/ticker_ids.npy"
        _write_npy(store, tokens_path, token_array[start:stop])
        _write_npy(store, ticker_ids_path, ticker_id_array[start:stop])
        partitions.append(
            {
                "partition_id": partition_id,
                "tokens_path": tokens_path,
                "ticker_ids_path": ticker_ids_path,
                "sequence_count": stop - start,
                "row_start": start,
                "row_stop": stop,
            }
        )
    metadata: dict[str, Any] = {
        "format": "mega-trading-numpy-token-v1",
        "tokens_path": root_path,
        "ticker_ids_path": root_path,
        "metadata_path": metadata_path,
        "sequence_count": int(token_array.shape[0]),
        "sequence_length": int(token_array.shape[1]),
        "tokens_dtype": str(token_array.dtype),
        "ticker_ids_dtype": str(ticker_id_array.dtype),
        "partitioned": True,
        "partition_rows": partition_rows,
        "partitions": partitions,
        "splits": split_metadata or {},
        "ticker_to_id": ticker_to_id,
        "id_to_ticker": {str(index): ticker for ticker, index in ticker_to_id.items()},
    }
    store.write_json(metadata_path, metadata)
    return metadata


def _split_metadata(
    sequences: list[dict[str, Any]],
    validation_fraction: float,
    backtest_fraction: float,
) -> dict[str, Any]:
    by_ticker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sequences:
        by_ticker[str(row["ticker"])].append(row)
    counts = Counter({ticker: len(rows) for ticker, rows in by_ticker.items()})
    split_counts: dict[str, dict[str, int]] = {}
    time_ranges: dict[str, dict[str, dict[str, Any]]] = {}
    totals = {"train": 0, "validation": 0, "backtest": 0}
    for ticker, count in sorted(counts.items()):
        backtest = _heldout_count(count, backtest_fraction)
        remaining = count - backtest
        validation = _heldout_count(remaining, validation_fraction)
        train = count - validation - backtest
        ticker_ranges = _ticker_split_time_ranges(by_ticker[ticker], train, validation, backtest)
        split_counts[ticker] = {
            "train": train,
            "validation": validation,
            "backtest": backtest,
        }
        time_ranges[ticker] = ticker_ranges
        totals["train"] += train
        totals["validation"] += validation
        totals["backtest"] += backtest
    return {
        "method": "per_ticker_time",
        "order": ["train", "validation", "backtest"],
        "validation_fraction": validation_fraction,
        "backtest_fraction": backtest_fraction,
        "counts": split_counts,
        "time_ranges": time_ranges,
        "totals": totals,
    }


def _ticker_split_time_ranges(
    rows: list[dict[str, Any]],
    train: int,
    validation: int,
    backtest: int,
) -> dict[str, dict[str, Any]]:
    starts = {
        "train": 0,
        "validation": train,
        "backtest": train + validation,
    }
    counts = {
        "train": train,
        "validation": validation,
        "backtest": backtest,
    }
    ranges: dict[str, dict[str, Any]] = {}
    for split, start in starts.items():
        count = counts[split]
        if count <= 0:
            ranges[split] = {"sequence_count": 0, "start_time": None, "end_time": None}
            continue
        split_rows = rows[start : start + count]
        ranges[split] = {
            "sequence_count": count,
            "start_time": str(split_rows[0]["start_time"]),
            "end_time": str(split_rows[-1]["end_time"]),
        }
    return ranges


def _heldout_count(count: int, fraction: float) -> int:
    if count <= 1 or fraction <= 0.0:
        return 0
    return min(max(1, int(count * fraction)), count - 1)


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
