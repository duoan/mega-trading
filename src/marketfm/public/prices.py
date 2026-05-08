"""Public historical price adapters."""

from __future__ import annotations

import csv
from dataclasses import asdict
from io import StringIO
from typing import Callable
from urllib.request import urlopen

from marketfm.ingest import IngestResult
from marketfm.schemas import Manifest, PriceRecord
from marketfm.store import ArtifactPaths, LocalObjectStore

FetchText = Callable[[str], str]


class StooqClient:
    base_url = "https://stooq.com/q/d/l/"

    def __init__(self, fetch_text: FetchText | None = None) -> None:
        self.fetch_text = fetch_text or _fetch_text

    def daily(self, ticker: str, start: str, end: str) -> list[dict[str, object]]:
        symbol = f"{ticker.lower()}.us"
        url = f"{self.base_url}?s={symbol}&d1={start.replace('-', '')}&d2={end.replace('-', '')}&i=d"
        text = self.fetch_text(url)
        rows: list[dict[str, object]] = []
        for row in csv.DictReader(StringIO(text)):
            if not row.get("Date") or row.get("Close") in (None, "N/D"):
                continue
            rows.append(
                {
                    "ticker": ticker.upper(),
                    "date": row["Date"],
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "adjusted_close": float(row["Close"]),
                    "volume": int(float(row["Volume"])),
                }
            )
        return rows


class StooqPriceIngestor:
    def __init__(self, store: LocalObjectStore, client: StooqClient) -> None:
        self.store = store
        self.client = client
        self.paths = ArtifactPaths()

    def ingest(self, tickers: list[str], start: str, end: str) -> IngestResult:
        raw_rows: list[dict[str, object]] = []
        prices: list[PriceRecord] = []
        for ticker in tickers:
            rows = self.client.daily(ticker, start, end)
            raw_rows.append({"ticker": ticker.upper(), "start": start, "end": end, "rows": rows})
            for row in rows:
                price_id = f"stooq-{row['ticker']}-{row['date']}"
                prices.append(
                    PriceRecord(
                        price_id=price_id,
                        ticker=str(row["ticker"]),
                        date=str(row["date"]),
                        adjusted_close=float(row["adjusted_close"]),
                        provider="stooq",
                        source_ids=[price_id],
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=int(row["volume"]),
                    )
                )

        bronze_path = self.paths.bronze("stooq", "daily_prices")
        silver_path = self.paths.silver("prices", "stooq")
        self.store.write_jsonl(bronze_path, raw_rows)
        self.store.write_jsonl(silver_path, [asdict(price) for price in prices])

        bronze_manifest = Manifest(
            manifest_id="stooq-daily-bronze",
            artifact_type="bronze",
            paths=[bronze_path],
            metadata={"source": "stooq_daily", "tickers": ",".join(tickers), "record_count": str(len(raw_rows))},
        )
        bronze_manifest_path = self.paths.manifest("ingest", "stooq-daily-bronze")
        self.store.write_manifest(bronze_manifest_path, bronze_manifest)

        normalization_manifest = Manifest(
            manifest_id="stooq-daily-normalized",
            artifact_type="silver",
            paths=[silver_path],
            metadata={
                "source": "stooq_daily",
                "source_manifest_id": bronze_manifest.manifest_id,
                "prices": str(len(prices)),
            },
        )
        normalization_manifest_path = self.paths.manifest("normalization", "stooq-daily-normalized")
        self.store.write_manifest(normalization_manifest_path, normalization_manifest)

        return IngestResult(
            bronze_manifest_path=bronze_manifest_path,
            normalization_manifest_path=normalization_manifest_path,
            normalized_counts={"prices": len(prices)},
            quality_summary={"quarantined_records": 0, "duplicate_records": 0},
        )


def _fetch_text(url: str) -> str:
    with urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8")
