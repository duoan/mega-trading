"""Direct data preparation for training-ready NumPy shards."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import math
import os
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, Iterable

import numpy as np

from mega_trading.config import BuildConfig
from mega_trading.core.schemas import Manifest
from mega_trading.core.store import LocalObjectStore
from mega_trading.data.fixtures import load_fixture_bundle
from mega_trading.data.ingest import BinanceTradesIngestRequest
from mega_trading.data.ingest_config import IngestPipelineConfig
from mega_trading.data.public.binance import (
    BINANCE_TRADES_BASE_URL,
    BinanceTradeArchive,
    count_trade_rows_in_archive,
    download_binance_trade_archives,
    iter_trade_fields_from_archive,
    iter_order_flow_event_dicts_for_symbol,
    iter_valid_trade_rows_from_archive,
    load_order_flow_from_archives,
    sample_trade_quantities_from_archive,
    _to_utc_datetime,
)
from mega_trading.events import BuildResult, EventBuilder, STREAM_CONTRACT, events_by_ticker_from_records, events_by_ticker_from_rows
from mega_trading.tokenizer import BOS_TOKEN, EOS_TOKEN, MarketEventTokenizer


def prepare_numpy_dataset(ingest_config: IngestPipelineConfig, build_config: BuildConfig) -> BuildResult:
    """Prepare training shards from fixture rows or cached Binance raw archives."""
    started_at = perf_counter()
    events_by_ticker: dict[str, list[dict[str, object]]] = {}
    for source in ingest_config.enabled_sources():
        source_started_at = perf_counter()
        if source.name == "fixture":
            rows = [asdict(row) for row in load_fixture_bundle().order_flow]
            source_events_by_ticker = events_by_ticker_from_rows(rows)
        elif source.name == "binance_trades":
            store = LocalObjectStore(Path(ingest_config.output_dir))
            request = BinanceTradesIngestRequest(
                symbols=source.tickers,
                start=str(source.start),
                end=str(source.end),
                frequency=source.frequency,
                download_workers=source.download_workers,
                process_workers=source.process_workers,
            )
            archives = download_binance_trade_archives(request, store.root, BINANCE_TRADES_BASE_URL)
            if build_config.streaming_prepare:
                _clean_intermediates(store, build_config)
                result = _prepare_streaming_binance_dataset(store, request, archives, build_config, source_started_at)
                print(
                    "direct prepare: "
                    f"streamed Binance archives as NumPy shards in {perf_counter() - started_at:.1f}s; "
                    f"metadata={result.numpy_metadata_path}"
                )
                return result
            loaded_at = perf_counter()
            events = load_order_flow_from_archives(request, archives)
            source_events_by_ticker = events_by_ticker_from_records(events, assume_sorted=True)
            print(
                "direct prepare: "
                f"cached {len(archives)} {source.name} archives in {loaded_at - source_started_at:.1f}s; "
                f"converted {len(events)} events in {perf_counter() - loaded_at:.1f}s"
            )
        else:
            raise ValueError(f"unsupported direct prepare source: {source.name}")
        _merge_events(events_by_ticker, source_events_by_ticker)

    store = LocalObjectStore(Path(ingest_config.output_dir))
    _clean_intermediates(store, build_config)
    event_count = sum(len(events) for events in events_by_ticker.values())
    result = EventBuilder(store, build_config).build_events(events_by_ticker)
    print(
        "direct prepare: "
        f"wrote {event_count} events as NumPy shards in {perf_counter() - started_at:.1f}s; "
        f"metadata={result.numpy_metadata_path}"
    )
    return result


def _clean_intermediates(store: LocalObjectStore, build_config: BuildConfig) -> None:
    store.delete_tree_if_exists("stage=02_normalized")
    store.delete_tree_if_exists(f"stage=04_corpus/mixture={build_config.mixture_name}")


def _merge_events(
    target: dict[str, list[dict[str, object]]],
    source: dict[str, list[dict[str, object]]],
) -> None:
    for ticker, events in source.items():
        target.setdefault(ticker, []).extend(events)


def _prepare_streaming_binance_dataset(
    store: LocalObjectStore,
    request: BinanceTradesIngestRequest,
    archives: list[BinanceTradeArchive],
    config: BuildConfig,
    started_at: float,
) -> BuildResult:
    archives_by_symbol = _archives_by_symbol(archives)
    if config.max_tickers is not None:
        selected = sorted(archives_by_symbol)[: config.max_tickers]
        archives_by_symbol = {ticker: archives_by_symbol[ticker] for ticker in selected}
    _log(
        "stream prepare: "
        f"cached {len(archives)} archives in {perf_counter() - started_at:.1f}s; "
        f"scanning {len(archives_by_symbol)} tickers with bounded memory"
    )

    stats_started_at = perf_counter()
    workers = _streaming_worker_count(request.process_workers, len(archives_by_symbol))
    _log(f"stream prepare: using {workers} CPU workers for read-only ZIP scans")
    ticker_counts, qty_baselines = _streaming_symbol_stats(
        request,
        archives_by_symbol,
        config.streaming_baseline_sample_rows,
        workers,
    )
    included_symbols = [
        symbol
        for symbol in sorted(archives_by_symbol)
        if ticker_counts.get(symbol, 0) >= config.min_events_per_ticker and symbol in qty_baselines
    ]
    if not included_symbols:
        raise ValueError("no Binance symbols had enough events for streaming prepare")
    archives_by_symbol = {symbol: archives_by_symbol[symbol] for symbol in included_symbols}
    ticker_counts = {symbol: ticker_counts[symbol] for symbol in included_symbols}
    _log(
        "stream prepare: "
        f"computed sampled baselines for {len(included_symbols)} tickers in {perf_counter() - stats_started_at:.1f}s"
    )

    fit_started_at = perf_counter()
    tokenizer_sample = list(
        _sample_streaming_events(
            request,
            archives_by_symbol,
            qty_baselines,
            config.streaming_tokenizer_sample_events,
            workers,
        )
    )
    tokenizer = MarketEventTokenizer.fit(
        tokenizer_sample,
        relative_price_bins=config.tokenizer_relative_price_bins,
        price_bins=config.tokenizer_price_bins,
        size_bins=config.tokenizer_size_bins,
        time_bins=config.tokenizer_time_bins,
        method=config.tokenizer_method,
        clip_quantile=config.tokenizer_clip_quantile,
    )
    tokenizer_path = f"datasets/mixture={config.mixture_name}/tokenizer.json"
    store.write_json(tokenizer_path, tokenizer.to_dict())
    _log(
        "stream prepare: "
        f"fit tokenizer from {len(tokenizer_sample)} sampled events in {perf_counter() - fit_started_at:.1f}s"
    )

    count_started_at = perf_counter()
    sequence_counts = {
        symbol: _sequence_count_from_event_count(event_count, config.block_size, config.stride)
        for symbol, event_count in ticker_counts.items()
    }
    sequence_counts = {symbol: count for symbol, count in sequence_counts.items() if count > 0}
    archives_by_symbol = {symbol: archives_by_symbol[symbol] for symbol in sorted(sequence_counts)}
    ticker_counts = {symbol: ticker_counts[symbol] for symbol in sorted(sequence_counts)}
    total_sequences = sum(sequence_counts.values())
    if total_sequences <= 0:
        raise ValueError("not enough streaming Binance events to build token sequences")
    _log(
        "stream prepare: "
        f"planned {total_sequences} token sequences in {perf_counter() - count_started_at:.1f}s"
    )

    write_started_at = perf_counter()
    numpy_metadata = _write_streaming_numpy_dataset(
        store,
        request,
        archives_by_symbol,
        qty_baselines,
        tokenizer,
        config,
        ticker_counts,
        sequence_counts,
        workers,
    )
    profile_path = f"datasets/mixture={config.mixture_name}/tokens-profile.json"
    manifest_path = f"manifests/build/{config.mixture_name}.json"
    profile = _streaming_profile(ticker_counts, sequence_counts, tokenizer, config, tokenizer_path, numpy_metadata)
    store.write_json(profile_path, profile)
    manifest_paths = [
        profile_path,
        tokenizer_path,
        str(numpy_metadata["tokens_path"]),
        str(numpy_metadata["ticker_ids_path"]),
        str(numpy_metadata["metadata_path"]),
    ]
    store.write_manifest(
        manifest_path,
        Manifest(
            manifest_id=f"{config.mixture_name}-build",
            artifact_type="event-token-build",
            paths=manifest_paths,
            metadata=profile,
        ),
    )
    _log(
        "stream prepare: "
        f"wrote streaming NumPy shards in {perf_counter() - write_started_at:.1f}s"
    )
    return BuildResult(
        profile_path=profile_path,
        manifest_path=manifest_path,
        tokenizer_path=tokenizer_path,
        numpy_tokens_path=str(numpy_metadata["tokens_path"]),
        numpy_ticker_ids_path=str(numpy_metadata["ticker_ids_path"]),
        numpy_metadata_path=str(numpy_metadata["metadata_path"]),
    )


def _archives_by_symbol(archives: list[BinanceTradeArchive]) -> dict[str, list[BinanceTradeArchive]]:
    grouped: dict[str, list[BinanceTradeArchive]] = defaultdict(list)
    for archive in archives:
        grouped[archive.symbol].append(archive)
    return {symbol: sorted(values, key=lambda archive: archive.partition) for symbol, values in sorted(grouped.items())}


def _streaming_symbol_stats(
    request: BinanceTradesIngestRequest,
    archives_by_symbol: dict[str, list[BinanceTradeArchive]],
    max_sample_rows: int,
    workers: int,
) -> tuple[dict[str, int], dict[str, float]]:
    ticker_row_counts: dict[str, int] = defaultdict(int)
    qty_samples: dict[str, list[float]] = defaultdict(list)
    max_archives_per_symbol = max((len(archives) for archives in archives_by_symbol.values()), default=1)
    per_archive_sample_rows = max(256, min(4096, max_sample_rows // max(max_archives_per_symbol, 1)))
    jobs = [
        (request.frequency, str(request.start), str(request.end), archive, per_archive_sample_rows)
        for archives in archives_by_symbol.values()
        for archive in archives
    ]
    results = (
        [_streaming_archive_stats_job(job) for job in jobs]
        if workers == 1
        else _parallel_map(_streaming_archive_stats_job, jobs, workers, "archive baseline scan")
    )
    for symbol, _partition, row_count, quantities in results:
        ticker_row_counts[symbol] += row_count
        sample = qty_samples[symbol]
        remaining = max(max_sample_rows - len(sample), 0)
        if remaining:
            sample.extend(quantities[:remaining])
    ticker_counts = {symbol: max(row_count - 1, 0) for symbol, row_count in ticker_row_counts.items()}
    baselines = {symbol: median(quantities) for symbol, quantities in qty_samples.items() if quantities}
    return ticker_counts, baselines


def _streaming_archive_stats_job(
    job: tuple[str, str, str, BinanceTradeArchive, int],
) -> tuple[str, str, int, list[float]]:
    frequency, start, end, archive, sample_rows = job
    start_dt = _to_utc_datetime(start)
    end_dt = _to_utc_datetime(end)
    if _archive_fully_covered(archive, frequency, start_dt, end_dt):
        return (
            archive.symbol,
            archive.partition,
            count_trade_rows_in_archive(archive.path),
            sample_trade_quantities_from_archive(archive.path, sample_rows),
        )
    row_count = 0
    quantities: list[float] = []
    for row in iter_valid_trade_rows_from_archive(archive.path, archive.symbol, start_dt, end_dt):
        row_count += 1
        if len(quantities) < sample_rows:
            quantities.append(float(row["qty"]))
    return archive.symbol, archive.partition, row_count, quantities


def _archive_fully_covered(
    archive: BinanceTradeArchive,
    frequency: str,
    start_dt: datetime,
    end_dt: datetime,
) -> bool:
    partition_start, partition_stop = _archive_partition_bounds(archive.partition, frequency)
    return start_dt <= partition_start and end_dt >= partition_stop - timedelta(microseconds=1)


def _archive_partition_bounds(partition: str, frequency: str) -> tuple[datetime, datetime]:
    if frequency == "daily":
        start = datetime.fromisoformat(partition).replace(tzinfo=timezone.utc)
        return start, start + timedelta(days=1)
    if frequency == "monthly":
        year, month = (int(part) for part in partition.split("-", 1))
        start = datetime(year, month, 1, tzinfo=timezone.utc)
        next_year = year + (1 if month == 12 else 0)
        next_month = 1 if month == 12 else month + 1
        return start, datetime(next_year, next_month, 1, tzinfo=timezone.utc)
    raise ValueError("frequency must be daily or monthly")


def _sample_streaming_events(
    request: BinanceTradesIngestRequest,
    archives_by_symbol: dict[str, list[BinanceTradeArchive]],
    qty_baselines: dict[str, float],
    max_events: int,
    workers: int,
) -> Iterable[dict[str, object]]:
    per_symbol_limit = max(1, max_events // max(len(archives_by_symbol), 1))
    jobs = [
        (request, symbol, archives, qty_baselines[symbol], per_symbol_limit)
        for symbol, archives in archives_by_symbol.items()
    ]
    results = (
        [_sample_streaming_events_job(job) for job in jobs]
        if workers == 1
        else _parallel_map(_sample_streaming_events_job, jobs, workers, "tokenizer sample")
    )
    for _symbol, rows in sorted(results, key=lambda item: item[0]):
        yield from rows


def _sample_streaming_events_job(
    job: tuple[BinanceTradesIngestRequest, str, list[BinanceTradeArchive], float, int],
) -> tuple[str, list[dict[str, object]]]:
    request, symbol, archives, qty_baseline, per_symbol_limit = job
    rows: list[dict[str, object]] = []
    for event in iter_order_flow_event_dicts_for_symbol(request, archives, qty_baseline):
        rows.append(event)
        if len(rows) >= per_symbol_limit:
            break
    return symbol, rows


def _sequence_count_from_event_count(event_count: int, block_size: int, stride: int) -> int:
    token_count = event_count + 2
    target_length = block_size + 1
    if token_count < target_length:
        return 0
    return ((token_count - target_length) // stride) + 1


def _iter_symbol_sequence_rows(
    ticker: str,
    events: Iterable[dict[str, object]],
    tokenizer: MarketEventTokenizer,
    block_size: int,
    stride: int,
) -> Iterable[dict[str, Any]]:
    target_length = block_size + 1
    tokens = [BOS_TOKEN]
    timestamps = ["bos"]
    for event in events:
        tokens.extend(tokenizer.encode_event(event))
        timestamps.append(str(event["timestamp"]))
        yield from _drain_sequences(ticker, tokens, timestamps, target_length, stride)
    tokens.append(EOS_TOKEN)
    timestamps.append("eos")
    yield from _drain_sequences(ticker, tokens, timestamps, target_length, stride)


def _drain_sequences(
    ticker: str,
    tokens: list[int],
    timestamps: list[str],
    target_length: int,
    stride: int,
) -> Iterable[dict[str, Any]]:
    while len(tokens) >= target_length:
        yield {
            "ticker": ticker,
            "start_time": timestamps[0],
            "end_time": timestamps[target_length - 1],
            "tokens": list(tokens[:target_length]),
        }
        del tokens[:stride]
        del timestamps[:stride]


def _write_streaming_numpy_dataset(
    store: LocalObjectStore,
    request: BinanceTradesIngestRequest,
    archives_by_symbol: dict[str, list[BinanceTradeArchive]],
    qty_baselines: dict[str, float],
    tokenizer: MarketEventTokenizer,
    config: BuildConfig,
    ticker_counts: dict[str, int],
    sequence_counts: dict[str, int],
    workers: int,
) -> dict[str, Any]:
    partition_rows = config.numpy_partition_rows or 65_536
    root_path = f"datasets/mixture={config.mixture_name}/numpy"
    metadata_path = f"datasets/mixture={config.mixture_name}/tokens-numpy.json"
    store.delete_if_exists(f"datasets/mixture={config.mixture_name}/tokens.npy")
    store.delete_if_exists(f"datasets/mixture={config.mixture_name}/ticker_ids.npy")
    store.delete_tree_if_exists(root_path)
    tickers = sorted(sequence_counts)
    ticker_to_id = {ticker: index for index, ticker in enumerate(tickers)}
    split_counts = _split_counts(sequence_counts, config.validation_fraction, config.backtest_fraction)
    totals = {"train": 0, "validation": 0, "backtest": 0}
    for counts in split_counts.values():
        for split in totals:
            totals[split] += counts[split]

    jobs = [
        (
            str(store.root),
            request,
            ticker,
            archives_by_symbol[ticker],
            qty_baselines[ticker],
            tokenizer,
            ticker_counts[ticker] + 2,
            root_path,
            ticker_to_id[ticker],
            sequence_counts[ticker],
        )
        for ticker in tickers
    ]
    results = (
        [_write_symbol_token_stream_job(job) for job in jobs]
        if workers == 1
        else _parallel_map(_write_symbol_token_stream_job, jobs, workers, "symbol token stream write")
    )
    results = sorted(results, key=lambda item: item["ticker"])
    partitions: list[dict[str, Any]] = []
    time_ranges = _empty_time_ranges(split_counts)
    row_offset = 0
    for result in results:
        sequence_count = int(result["sequence_count"])
        adjusted = dict(result)
        adjusted["row_start"] = row_offset
        adjusted["row_stop"] = row_offset + sequence_count
        partitions.append(adjusted)
        row_offset += sequence_count

    metadata: dict[str, Any] = {
        "format": "mega-trading-numpy-token-v1",
        "storage": "token_stream",
        "tokens_path": root_path,
        "ticker_ids_path": root_path,
        "metadata_path": metadata_path,
        "sequence_count": row_offset,
        "sequence_length": config.block_size + 1,
        "stride": config.stride,
        "tokens_dtype": "int32",
        "ticker_ids_dtype": "int32",
        "partitioned": True,
        "partition_rows": partition_rows,
        "partitions": partitions,
        "splits": {
            "method": "per_ticker_time",
            "order": ["train", "validation", "backtest"],
            "validation_fraction": config.validation_fraction,
            "backtest_fraction": config.backtest_fraction,
            "counts": split_counts,
            "time_ranges": time_ranges,
            "totals": totals,
        },
        "ticker_to_id": ticker_to_id,
        "id_to_ticker": {str(index): ticker for ticker, index in ticker_to_id.items()},
    }
    store.write_json(metadata_path, metadata)
    return metadata


def _write_symbol_token_stream_job(
    job: tuple[
        str,
        BinanceTradesIngestRequest,
        str,
        list[BinanceTradeArchive],
        float,
        MarketEventTokenizer,
        int,
        str,
        int,
        int,
    ],
) -> dict[str, Any]:
    (
        root,
        request,
        ticker,
        archives,
        qty_baseline,
        tokenizer,
        token_count,
        root_path,
        ticker_id,
        sequence_count,
    ) = job
    root_dir = Path(root)
    partition_id = f"symbol={ticker}/stream"
    tokens_path = f"{root_path}/{partition_id}/tokens.npy"
    target = root_dir / tokens_path
    target.parent.mkdir(parents=True, exist_ok=True)
    token_stream = np.lib.format.open_memmap(target, mode="w+", dtype=np.int32, shape=(token_count,))
    offset = 0
    token_stream[offset] = BOS_TOKEN
    offset += 1
    for token in _iter_fast_symbol_tokens(request, archives, qty_baseline, tokenizer):
        if offset >= token_count - 1:
            break
        token_stream[offset] = token
        offset += 1
    if offset < token_count:
        token_stream[offset] = EOS_TOKEN
        offset += 1
    if offset < token_count:
        token_stream[offset:] = EOS_TOKEN
    token_stream.flush()
    return {
        "ticker": ticker,
        "ticker_id": ticker_id,
        "partition_id": partition_id,
        "tokens_path": tokens_path,
        "sequence_count": sequence_count,
        "token_count": token_count,
    }


def _iter_fast_symbol_tokens(
    request: BinanceTradesIngestRequest,
    archives: list[BinanceTradeArchive],
    qty_baseline: float,
    tokenizer: MarketEventTokenizer,
) -> Iterable[int]:
    start_seconds = _to_utc_datetime(str(request.start)).timestamp()
    end_seconds = _to_utc_datetime(str(request.end)).timestamp()
    previous_price: float | None = None
    previous_time: float | None = None
    for archive in sorted(archives, key=lambda item: item.partition):
        for _trade_id, price, qty, raw_time, is_buyer_maker in iter_trade_fields_from_archive(archive.path):
            timestamp = _binance_timestamp_seconds(raw_time)
            if timestamp < start_seconds or timestamp > end_seconds or qty <= 0.0 or price <= 0.0:
                continue
            if previous_price is None or previous_time is None:
                previous_price = price
                previous_time = timestamp
                continue
            relative_price_bps = 10_000.0 * math.log(price / max(previous_price, 1e-12))
            yield tokenizer.encode_features(
                action="delete",
                side="sell" if is_buyer_maker else "buy",
                relative_price_bps=float(relative_price_bps),
                price_depth_bps=abs(float(relative_price_bps)),
                size=qty / max(qty_baseline, 1e-12),
                interarrival_seconds=max(timestamp - previous_time, 1e-6),
            )
            previous_price = price
            previous_time = timestamp


def _binance_timestamp_seconds(value: int) -> float:
    divisor = 1_000_000 if value >= 10**15 else 1_000
    return value / divisor


def _split_counts(
    sequence_counts: dict[str, int],
    validation_fraction: float,
    backtest_fraction: float,
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for ticker, count in sequence_counts.items():
        backtest = _heldout_count(count, backtest_fraction)
        remaining = count - backtest
        validation = _heldout_count(remaining, validation_fraction)
        result[ticker] = {
            "train": count - validation - backtest,
            "validation": validation,
            "backtest": backtest,
        }
    return result


def _heldout_count(count: int, fraction: float) -> int:
    if count <= 1 or fraction <= 0.0:
        return 0
    return min(max(1, int(count * fraction)), count - 1)


def _empty_time_ranges(split_counts: dict[str, dict[str, int]]) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        ticker: {
            split: {"sequence_count": counts[split], "start_time": None, "end_time": None}
            for split in ("train", "validation", "backtest")
        }
        for ticker, counts in split_counts.items()
    }


def _split_for_index(index: int, counts: dict[str, int]) -> str | None:
    train = counts["train"]
    validation = counts["validation"]
    backtest = counts["backtest"]
    if index < train:
        return "train"
    if index < train + validation:
        return "validation"
    if index < train + validation + backtest:
        return "backtest"
    return None


def _update_time_range(target: dict[str, Any], row: dict[str, Any]) -> None:
    if target["start_time"] is None:
        target["start_time"] = str(row["start_time"])
    target["end_time"] = str(row["end_time"])


def _streaming_profile(
    ticker_counts: dict[str, int],
    sequence_counts: dict[str, int],
    tokenizer: MarketEventTokenizer,
    config: BuildConfig,
    tokenizer_path: str,
    numpy_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stream_contract": STREAM_CONTRACT,
        "feature_contract": "paper-order-flow-v1",
        "feature_order": ["action", "side", "relative_price", "price_depth", "size", "time"],
        "mixture": config.mixture_name,
        "source": config.source,
        "event_count": sum(ticker_counts.values()),
        "sequence_count": sum(sequence_counts.values()),
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
        "streaming_prepare": True,
        "streaming_tokenizer_sample_events": config.streaming_tokenizer_sample_events,
        "streaming_baseline_sample_rows": config.streaming_baseline_sample_rows,
        "numpy_dataset": numpy_metadata,
    }


def _write_npy(store: LocalObjectStore, path: str, value: np.ndarray) -> None:
    _write_npy_path(store.root / path, value)


def _write_npy_path(target: Path, value: np.ndarray) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, value)


def _streaming_worker_count(requested_workers: int, task_count: int) -> int:
    if task_count <= 0:
        return 1
    if requested_workers > 0:
        return max(1, min(requested_workers, task_count))
    return max(1, min(os.cpu_count() or 1, task_count))


def _parallel_map(function, jobs: list[Any], workers: int, label: str) -> list[Any]:
    results: list[Any] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(function, job) for job in jobs]
        total = len(futures)
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            _log(f"stream prepare: {label} finished {index}/{total}: {_result_symbol(result)}")
            results.append(result)
    return results


def _result_symbol(result: Any) -> str:
    if isinstance(result, dict) and "ticker" in result:
        return str(result["ticker"])
    if isinstance(result, tuple) and result:
        return str(result[0])
    return "unknown"


def _log(message: str) -> None:
    print(message, flush=True)
