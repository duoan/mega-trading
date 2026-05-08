"""Command line interface for MarketFM Forge."""

from __future__ import annotations

import argparse
from pathlib import Path

from marketfm.public.prices import YahooChartClient, YahooPriceIngestor
from marketfm.public.sec import SecClient, SecCompanyFactsIngestor
from marketfm.store import LocalObjectStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="marketfm",
        description="MarketFM Forge data and training infrastructure CLI.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="print the package version and exit",
    )
    subparsers = parser.add_subparsers(dest="command")
    ingest_public = subparsers.add_parser("ingest-public", help="ingest public SEC fundamentals and Yahoo prices")
    ingest_public.add_argument("--tickers", required=True, help="comma-separated ticker symbols")
    ingest_public.add_argument("--start", required=True, help="price start date YYYY-MM-DD")
    ingest_public.add_argument("--end", required=True, help="price end date YYYY-MM-DD")
    ingest_public.add_argument("--out", default=".marketfm/public", help="artifact output directory")
    ingest_public.add_argument("--sec-user-agent", required=True, help="SEC-compliant User-Agent, including contact email")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        from marketfm import __version__

        print(__version__)
    elif args.command == "ingest-public":
        tickers = [ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()]
        store = LocalObjectStore(Path(args.out))
        SecCompanyFactsIngestor(store, SecClient(user_agent=args.sec_user_agent)).ingest(tickers)
        YahooPriceIngestor(store, YahooChartClient()).ingest(tickers, args.start, args.end)
        print(f"wrote public artifacts for {','.join(tickers)} to {args.out}")
    return 0
