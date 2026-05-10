"""Direct data preparation for training-ready NumPy shards."""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
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
    download_binance_trade_archives,
    iter_order_flow_event_dicts_for_symbol,
    iter_valid_trade_rows_from_archive,
    load_order_flow_from_archives,
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
    tokenizer_path = f"stage=05_shards/mixture={config.mixture_name}/tokenizer.json"
    store.write_json(tokenizer_path, tokenizer.to_dict())
    _log(
        "stream prepare: "
        f"fit tokenizer from {len(tokenizer_sample)} sampled events in {perf_counter() - fit_started_at:.1f}s"
    )

    count_started_at = perf_counter()
    sequence_counts = _count_streaming_sequences(request, archives_by_symbol, qty_baselines, tokenizer, config, workers)
    sequence_counts = {symbol: count for symbol, count in sequence_counts.items() if count > 0}
    archives_by_symbol = {symbol: archives_by_symbol[symbol] for symbol in sorted(sequence_counts)}
    ticker_counts = {symbol: ticker_counts[symbol] for symbol in sorted(sequence_counts)}
    total_sequences = sum(sequence_counts.values())
    if total_sequences <= 0:
        raise ValueError("not enough streaming Binance events to build token sequences")
    _log(
        "stream prepare: "
        f"counted {total_sequences} token sequences in {perf_counter() - count_started_at:.1f}s"
    )

    write_started_at = perf_counter()
    numpy_metadata = _write_streaming_numpy_dataset(
        store,
        request,
        archives_by_symbol,
        qty_baselines,
        tokenizer,
        config,
        sequence_counts,
    )
    profile_path = f"stage=05_shards/mixture={config.mixture_name}/tokens-profile.json"
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
    ticker_counts: dict[str, int] = {}
    baselines: dict[str, float] = {}
    jobs = [(request, symbol, archives, max_sample_rows) for symbol, archives in archives_by_symbol.items()]
    results = (
        [_streaming_symbol_stats_job(job) for job in jobs]
        if workers == 1
        else _parallel_map(_streaming_symbol_stats_job, jobs, workers, "baseline scan")
    )
    for symbol, row_count, baseline in results:
        ticker_counts[symbol] = max(row_count - 1, 0)
        if baseline is not None:
            baselines[symbol] = baseline
    return ticker_counts, baselines


def _streaming_symbol_stats_job(
    job: tuple[BinanceTradesIngestRequest, str, list[BinanceTradeArchive], int],
) -> tuple[str, int, float | None]:
    request, symbol, archives, max_sample_rows = job
    start_dt = _to_utc_datetime(request.start)
    end_dt = _to_utc_datetime(request.end)
    row_count = 0
    qty_sample: list[float] = []
    for archive in archives:
        for row in iter_valid_trade_rows_from_archive(archive.path, symbol, start_dt, end_dt):
            row_count += 1
            if len(qty_sample) < max_sample_rows:
                qty_sample.append(float(row["qty"]))
    return symbol, row_count, median(qty_sample) if qty_sample else None


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


def _count_streaming_sequences(
    request: BinanceTradesIngestRequest,
    archives_by_symbol: dict[str, list[BinanceTradeArchive]],
    qty_baselines: dict[str, float],
    tokenizer: MarketEventTokenizer,
    config: BuildConfig,
    workers: int,
) -> dict[str, int]:
    jobs = [
        (request, symbol, archives, qty_baselines[symbol], tokenizer, config.block_size, config.stride)
        for symbol, archives in archives_by_symbol.items()
    ]
    results = (
        [_count_streaming_sequences_job(job) for job in jobs]
        if workers == 1
        else _parallel_map(_count_streaming_sequences_job, jobs, workers, "sequence count")
    )
    return dict(results)


def _count_streaming_sequences_job(
    job: tuple[BinanceTradesIngestRequest, str, list[BinanceTradeArchive], float, MarketEventTokenizer, int, int],
) -> tuple[str, int]:
    request, symbol, archives, qty_baseline, tokenizer, block_size, stride = job
    count = sum(
        1
        for _row in _iter_symbol_sequence_rows(
            symbol,
            iter_order_flow_event_dicts_for_symbol(request, archives, qty_baseline),
            tokenizer,
            block_size,
            stride,
        )
    )
    return symbol, count


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
    sequence_counts: dict[str, int],
) -> dict[str, Any]:
    partition_rows = config.numpy_partition_rows or 65_536
    root_path = f"stage=05_shards/mixture={config.mixture_name}/numpy"
    metadata_path = f"stage=05_shards/mixture={config.mixture_name}/tokens-numpy.json"
    store.delete_if_exists(f"stage=05_shards/mixture={config.mixture_name}/tokens.npy")
    store.delete_if_exists(f"stage=05_shards/mixture={config.mixture_name}/ticker_ids.npy")
    store.delete_tree_if_exists(root_path)
    tickers = sorted(sequence_counts)
    ticker_to_id = {ticker: index for index, ticker in enumerate(tickers)}
    split_counts = _split_counts(sequence_counts, config.validation_fraction, config.backtest_fraction)
    time_ranges = _empty_time_ranges(split_counts)
    totals = {"train": 0, "validation": 0, "backtest": 0}
    for counts in split_counts.values():
        for split in totals:
            totals[split] += counts[split]

    sequence_length = config.block_size + 1
    token_buffer = np.empty((partition_rows, sequence_length), dtype=np.int64)
    ticker_buffer = np.empty((partition_rows,), dtype=np.int32)
    partitions: list[dict[str, Any]] = []
    row_offset = 0
    fill = 0
    seen_by_ticker: Counter[str] = Counter()

    def flush() -> None:
        nonlocal fill, row_offset
        if fill == 0:
            return
        partition_index = len(partitions)
        partition_id = f"part-{partition_index:05d}"
        tokens_path = f"{root_path}/{partition_id}/tokens.npy"
        ticker_ids_path = f"{root_path}/{partition_id}/ticker_ids.npy"
        _write_npy(store, tokens_path, token_buffer[:fill])
        _write_npy(store, ticker_ids_path, ticker_buffer[:fill])
        partitions.append(
            {
                "partition_id": partition_id,
                "tokens_path": tokens_path,
                "ticker_ids_path": ticker_ids_path,
                "sequence_count": fill,
                "row_start": row_offset,
                "row_stop": row_offset + fill,
            }
        )
        row_offset += fill
        fill = 0

    for ticker in tickers:
        for row in _iter_symbol_sequence_rows(
            ticker,
            iter_order_flow_event_dicts_for_symbol(request, archives_by_symbol[ticker], qty_baselines[ticker]),
            tokenizer,
            config.block_size,
            config.stride,
        ):
            index = seen_by_ticker[ticker]
            seen_by_ticker[ticker] += 1
            split = _split_for_index(index, split_counts[ticker])
            if split is not None:
                _update_time_range(time_ranges[ticker][split], row)
            token_buffer[fill, :] = np.asarray(row["tokens"], dtype=np.int64)
            ticker_buffer[fill] = ticker_to_id[ticker]
            fill += 1
            if fill >= partition_rows:
                flush()
        _log(f"stream prepare: wrote sequences for {ticker}: {seen_by_ticker[ticker]}")
    flush()

    metadata: dict[str, Any] = {
        "format": "mega-trading-numpy-token-v1",
        "tokens_path": root_path,
        "ticker_ids_path": root_path,
        "metadata_path": metadata_path,
        "sequence_count": row_offset,
        "sequence_length": sequence_length,
        "tokens_dtype": "int64",
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
    target = store.root / path
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
    if isinstance(result, tuple) and result:
        return str(result[0])
    return "unknown"


def _log(message: str) -> None:
    print(message, flush=True)
