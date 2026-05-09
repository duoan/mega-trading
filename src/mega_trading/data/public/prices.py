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

from mega_trading.core.schemas import Manifest, PriceRecord
from mega_trading.core.store import ArtifactPaths, LocalObjectStore
from mega_trading.data.ingest import Ingestor, IngestResult, PriceIngestRequest
from mega_trading.data.lance_store import LanceTableStore, LanceTables

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


class StooqPriceIngestor(Ingestor[PriceIngestRequest]):
    def __init__(self, store: LocalObjectStore, client: StooqClient, table_store: LanceTableStore | None = None) -> None:
        self.store = store
        self.client = client
        self.table_store = table_store
        self.tables = LanceTables()
        self.paths = ArtifactPaths()

    def ingest(self, request: PriceIngestRequest) -> IngestResult:
        tickers = list(request.tickers)
        start = request.start
        end = request.end
        raw_rows: list[dict[str, object]] = []
        prices: list[PriceRecord] = []
        total = len(tickers)
        for index, ticker in enumerate(tickers, start=1):
            if index == 1 or index % 25 == 0 or index == total:
                print(f"stooq_prices ingest progress: {index}/{total}")
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

        raw_path = self.paths.raw("stooq", "daily_prices")
        normalized_path = self.paths.normalized("prices", "stooq")
        self.store.write_jsonl(raw_path, raw_rows)
        price_rows = [asdict(price) for price in prices]
        self.store.write_jsonl(normalized_path, price_rows)
        if self.table_store:
            self.table_store.write_table(self.tables.normalized("prices", "stooq"), price_rows)

        raw_manifest = Manifest(
            manifest_id="stooq-daily-raw",
            artifact_type="raw",
            paths=[raw_path],
            metadata={"source": "stooq_daily", "tickers": ",".join(tickers), "record_count": str(len(raw_rows))},
        )
        raw_manifest_path = self.paths.manifest("ingest", "stooq-daily-raw")
        self.store.write_manifest(raw_manifest_path, raw_manifest)

        normalization_manifest = Manifest(
            manifest_id="stooq-daily-normalized",
            artifact_type="normalized",
            paths=[normalized_path],
            metadata={
                "source": "stooq_daily",
                "source_manifest_id": raw_manifest.manifest_id,
                "prices": str(len(prices)),
            },
        )
        normalization_manifest_path = self.paths.manifest("normalization", "stooq-daily-normalized")
        self.store.write_manifest(normalization_manifest_path, normalization_manifest)

        return IngestResult(
            raw_manifest_path=raw_manifest_path,
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


class YahooPriceIngestor(Ingestor[PriceIngestRequest]):
    def __init__(self, store: LocalObjectStore, client: YahooChartClient, table_store: LanceTableStore | None = None) -> None:
        self.store = store
        self.client = client
        self.table_store = table_store
        self.tables = LanceTables()
        self.paths = ArtifactPaths()

    def ingest(self, request: PriceIngestRequest) -> IngestResult:
        tickers = list(request.tickers)
        start = request.start
        end = request.end
        raw_rows: list[dict[str, object]] = []
        prices: list[PriceRecord] = []
        total = len(tickers)
        for index, ticker in enumerate(tickers, start=1):
            if index == 1 or index % 25 == 0 or index == total:
                print(f"yahoo_prices ingest progress: {index}/{total}")
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

        raw_path = self.paths.raw("yahoo", "daily_prices")
        normalized_path = self.paths.normalized("prices", "yahoo")
        self.store.write_jsonl(raw_path, raw_rows)
        price_rows = [asdict(price) for price in prices]
        self.store.write_jsonl(normalized_path, price_rows)
        if self.table_store:
            self.table_store.write_table(self.tables.normalized("prices", "yahoo"), price_rows)

        raw_manifest = Manifest(
            manifest_id="yahoo-daily-raw",
            artifact_type="raw",
            paths=[raw_path],
            metadata={"source": "yahoo_chart", "tickers": ",".join(tickers), "record_count": str(len(raw_rows))},
        )
        raw_manifest_path = self.paths.manifest("ingest", "yahoo-daily-raw")
        self.store.write_manifest(raw_manifest_path, raw_manifest)

        normalization_manifest = Manifest(
            manifest_id="yahoo-daily-normalized",
            artifact_type="normalized",
            paths=[normalized_path],
            metadata={
                "source": "yahoo_chart",
                "source_manifest_id": raw_manifest.manifest_id,
                "prices": str(len(prices)),
            },
        )
        normalization_manifest_path = self.paths.manifest("normalization", "yahoo-daily-normalized")
        self.store.write_manifest(normalization_manifest_path, normalization_manifest)

        return IngestResult(
            raw_manifest_path=raw_manifest_path,
            normalization_manifest_path=normalization_manifest_path,
            normalized_counts={"prices": len(prices)},
            quality_summary={"quarantined_records": 0, "duplicate_records": 0},
        )


def _fetch_text(url: str) -> str:
    context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(url, timeout=30, context=context) as response:
        return response.read().decode("utf-8")


def _fetch_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": "Mega-Trading"})
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
