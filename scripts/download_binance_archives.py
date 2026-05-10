"""Download Binance public trade archives to the local raw cache."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from mega_trading.data.ingest import BinanceTradesIngestRequest
from mega_trading.data.ingest_config import load_ingest_config
from mega_trading.data.public.binance import BINANCE_TRADES_BASE_URL, download_binance_trade_archives


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Binance trade ZIP archives without processing them.")
    parser.add_argument("--ingest-config", required=True, help="path to source TOML config")
    args = parser.parse_args()

    config = load_ingest_config(Path(args.ingest_config))
    output_dir = Path(config.output_dir)
    for source in config.enabled_sources():
        if source.name != "binance_trades":
            continue
        started_at = perf_counter()
        request = BinanceTradesIngestRequest(
            symbols=source.tickers,
            start=str(source.start),
            end=str(source.end),
            frequency=source.frequency,
            download_workers=source.download_workers,
            process_workers=source.process_workers,
        )
        archives = download_binance_trade_archives(request, output_dir, BINANCE_TRADES_BASE_URL)
        print(
            "download binance: "
            f"cached {len(archives)} archives under {output_dir / 'stage=01_raw'} "
            f"in {perf_counter() - started_at:.1f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
