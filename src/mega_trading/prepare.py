"""Direct data preparation for training-ready NumPy shards."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from mega_trading.config import BuildConfig
from mega_trading.core.store import LocalObjectStore
from mega_trading.data.fixtures import load_fixture_bundle
from mega_trading.data.ingest import BinanceTradesIngestRequest
from mega_trading.data.ingest_config import IngestPipelineConfig
from mega_trading.data.public.binance import (
    BINANCE_TRADES_BASE_URL,
    download_binance_trade_archives,
    load_order_flow_from_archives,
)
from mega_trading.events import BuildResult, EventBuilder, events_by_ticker_from_rows


def prepare_numpy_dataset(ingest_config: IngestPipelineConfig, build_config: BuildConfig) -> BuildResult:
    """Prepare training shards from fixture rows or cached Binance raw archives."""
    started_at = perf_counter()
    event_rows: list[dict[str, object]] = []
    for source in ingest_config.enabled_sources():
        source_started_at = perf_counter()
        if source.name == "fixture":
            rows = [asdict(row) for row in load_fixture_bundle().order_flow]
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
            loaded_at = perf_counter()
            events = load_order_flow_from_archives(request, archives)
            rows = [asdict(event) for event in events]
            print(
                "direct prepare: "
                f"cached {len(archives)} {source.name} archives in {loaded_at - source_started_at:.1f}s; "
                f"converted {len(rows)} events in {perf_counter() - loaded_at:.1f}s"
            )
        else:
            raise ValueError(f"unsupported direct prepare source: {source.name}")
        event_rows.extend(rows)

    store = LocalObjectStore(Path(ingest_config.output_dir))
    _clean_intermediates(store, build_config)
    result = EventBuilder(store, build_config).build_events(events_by_ticker_from_rows(event_rows))
    print(
        "direct prepare: "
        f"wrote {len(event_rows)} events as NumPy shards in {perf_counter() - started_at:.1f}s; "
        f"metadata={result.numpy_metadata_path}"
    )
    return result


def _clean_intermediates(store: LocalObjectStore, build_config: BuildConfig) -> None:
    store.delete_tree_if_exists("stage=02_normalized")
    store.delete_tree_if_exists(f"stage=04_corpus/mixture={build_config.mixture_name}")
