"""Public historical price adapters."""

from __future__ import annotations

import csv
import json
import ssl
from dataclasses import asdict
from datetime import datetime, timezone
from io import StringIO
from typing import Callable
from urllib.request import Request, urlopen

import certifi

from marketfm.core.schemas import Manifest, PriceRecord
from marketfm.core.store import ArtifactPaths, LocalObjectStore
from marketfm.data.ingest import IngestResult
from marketfm.data.lance_store import LanceTableStore, LanceTables

FetchText = Callable[[str], str]
FetchJson = Callable[[str], dict]


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
    def __init__(self, store: LocalObjectStore, client: StooqClient, table_store: LanceTableStore | None = None) -> None:
        self.store = store
        self.client = client
        self.table_store = table_store
        self.tables = LanceTables()
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
        price_rows = [asdict(price) for price in prices]
        self.store.write_jsonl(silver_path, price_rows)
        if self.table_store:
            self.table_store.write_table(self.tables.silver("prices", "stooq"), price_rows)

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


class YahooChartClient:
    base_url = "https://query1.finance.yahoo.com/v8/finance/chart"

    def __init__(self, fetch_json: FetchJson | None = None) -> None:
        self.fetch_json = fetch_json or _fetch_json

    def daily(self, ticker: str, start: str, end: str) -> list[dict[str, object]]:
        period1 = _to_unix(start)
        period2 = _to_unix(end) + 86_400
        symbol = ticker.upper()
        url = (
            f"{self.base_url}/{symbol}?period1={period1}&period2={period2}"
            "&interval=1d&events=history&includeAdjustedClose=true"
        )
        payload = self.fetch_json(url)
        result = payload.get("chart", {}).get("result", [])
        if not result:
            return []
        data = result[0]
        timestamps = data.get("timestamp", [])
        quote = data.get("indicators", {}).get("quote", [{}])[0]
        adjclose = data.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose", [])
        rows: list[dict[str, object]] = []
        for index, timestamp in enumerate(timestamps):
            close = _at(quote.get("close", []), index)
            if close is None:
                continue
            rows.append(
                {
                    "ticker": symbol,
                    "date": datetime.fromtimestamp(int(timestamp), tz=timezone.utc).date().isoformat(),
                    "open": _at(quote.get("open", []), index),
                    "high": _at(quote.get("high", []), index),
                    "low": _at(quote.get("low", []), index),
                    "close": close,
                    "adjusted_close": _at(adjclose, index) or close,
                    "volume": int(_at(quote.get("volume", []), index) or 0),
                }
            )
        return rows


class YahooPriceIngestor:
    def __init__(self, store: LocalObjectStore, client: YahooChartClient, table_store: LanceTableStore | None = None) -> None:
        self.store = store
        self.client = client
        self.table_store = table_store
        self.tables = LanceTables()
        self.paths = ArtifactPaths()

    def ingest(self, tickers: list[str], start: str, end: str) -> IngestResult:
        raw_rows: list[dict[str, object]] = []
        prices: list[PriceRecord] = []
        for ticker in tickers:
            rows = self.client.daily(ticker, start, end)
            raw_rows.append({"ticker": ticker.upper(), "start": start, "end": end, "rows": rows})
            for row in rows:
                price_id = f"yahoo-{row['ticker']}-{row['date']}"
                prices.append(
                    PriceRecord(
                        price_id=price_id,
                        ticker=str(row["ticker"]),
                        date=str(row["date"]),
                        adjusted_close=float(row["adjusted_close"]),
                        provider="yahoo",
                        source_ids=[price_id],
                        open=_optional_float(row["open"]),
                        high=_optional_float(row["high"]),
                        low=_optional_float(row["low"]),
                        close=_optional_float(row["close"]),
                        volume=int(row["volume"]),
                    )
                )

        bronze_path = self.paths.bronze("yahoo", "daily_prices")
        silver_path = self.paths.silver("prices", "yahoo")
        self.store.write_jsonl(bronze_path, raw_rows)
        price_rows = [asdict(price) for price in prices]
        self.store.write_jsonl(silver_path, price_rows)
        if self.table_store:
            self.table_store.write_table(self.tables.silver("prices", "yahoo"), price_rows)

        bronze_manifest = Manifest(
            manifest_id="yahoo-daily-bronze",
            artifact_type="bronze",
            paths=[bronze_path],
            metadata={"source": "yahoo_chart", "tickers": ",".join(tickers), "record_count": str(len(raw_rows))},
        )
        bronze_manifest_path = self.paths.manifest("ingest", "yahoo-daily-bronze")
        self.store.write_manifest(bronze_manifest_path, bronze_manifest)

        normalization_manifest = Manifest(
            manifest_id="yahoo-daily-normalized",
            artifact_type="silver",
            paths=[silver_path],
            metadata={
                "source": "yahoo_chart",
                "source_manifest_id": bronze_manifest.manifest_id,
                "prices": str(len(prices)),
            },
        )
        normalization_manifest_path = self.paths.manifest("normalization", "yahoo-daily-normalized")
        self.store.write_manifest(normalization_manifest_path, normalization_manifest)

        return IngestResult(
            bronze_manifest_path=bronze_manifest_path,
            normalization_manifest_path=normalization_manifest_path,
            normalized_counts={"prices": len(prices)},
            quality_summary={"quarantined_records": 0, "duplicate_records": 0},
        )


def _fetch_text(url: str) -> str:
    context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(url, timeout=30, context=context) as response:
        return response.read().decode("utf-8")


def _fetch_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": "MarketFM Forge"})
    context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(request, timeout=30, context=context) as response:
        return json.loads(response.read().decode("utf-8"))


def _to_unix(date: str) -> int:
    return int(datetime.fromisoformat(date).replace(tzinfo=timezone.utc).timestamp())


def _at(values: list, index: int) -> float | int | None:
    if index >= len(values):
        return None
    return values[index]


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)
