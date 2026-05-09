"""OHLCV-to-order-flow adapter for the paper-style feature contract."""

from __future__ import annotations

import math
import ssl
from dataclasses import asdict
from datetime import datetime, timezone
from statistics import median
from tempfile import NamedTemporaryFile
from typing import Callable
from urllib.request import Request, urlopen

import certifi
import pyarrow as pa
import pyarrow.parquet as pq

from mega_trading.core.schemas import Manifest, OrderFlowEventRecord
from mega_trading.core.store import ArtifactPaths, LocalObjectStore
from mega_trading.data.ingest import IngestResult, Ingestor, OhlcvIngestRequest
from mega_trading.data.lance_store import LanceTableStore, LanceTables

FetchParquetTable = Callable[[str], pa.Table]


class HuggingFaceOhlcvClient:
    """Fetch minute OHLCV bars used only as a public proxy for paper features."""

    base_url = "https://huggingface.co/datasets/mito0o852/OHLCV-1m/resolve/main/data"

    def __init__(self, fetch_table: FetchParquetTable | None = None) -> None:
        self.fetch_table = fetch_table or _fetch_parquet_table

    def minute(self, tickers: tuple[str, ...], start: str, end: str) -> list[dict[str, object]]:
        ticker_set = {ticker.upper() for ticker in tickers}
        start_dt = _to_utc_datetime(start)
        end_dt = _to_utc_datetime(end)
        rows: list[dict[str, object]] = []
        for month in _month_range(start_dt, end_dt):
            table = self.fetch_table(f"{self.base_url}/ohlcv_{month}.parquet")
            columns = table.select(["timestamp", "open", "high", "low", "close", "volume", "ticker"]).to_pydict()
            for index, ticker_value in enumerate(columns["ticker"]):
                ticker = str(ticker_value).upper()
                if ticker not in ticker_set:
                    continue
                timestamp = _timestamp_to_utc(columns["timestamp"][index])
                if timestamp < start_dt or timestamp > end_dt:
                    continue
                close = columns["close"][index]
                volume = columns["volume"][index]
                if close is None or volume is None or float(volume) <= 0.0:
                    continue
                rows.append(
                    {
                        "ticker": ticker,
                        "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
                        "open": _optional_float(columns["open"][index]),
                        "high": _optional_float(columns["high"][index]),
                        "low": _optional_float(columns["low"][index]),
                        "close": float(close),
                        "volume": float(volume),
                    }
                )
        return sorted(rows, key=lambda row: (str(row["ticker"]), str(row["timestamp"])))


class HuggingFaceOhlcvIngestor(Ingestor[OhlcvIngestRequest]):
    """Map minute OHLCV bars into action/side/depth/size/interarrival events."""

    def __init__(
        self,
        store: LocalObjectStore,
        client: HuggingFaceOhlcvClient,
        table_store: LanceTableStore | None = None,
    ) -> None:
        self.store = store
        self.client = client
        self.table_store = table_store
        self.tables = LanceTables()
        self.paths = ArtifactPaths()

    def ingest(self, request: OhlcvIngestRequest) -> IngestResult:
        raw_rows = self.client.minute(request.tickers, request.start, request.end)
        events = _ohlcv_rows_to_order_flow(raw_rows)
        event_rows = [asdict(event) for event in events]

        raw_path = self.paths.raw("hf_ohlcv_1m", "ohlcv")
        normalized_path = self.paths.normalized("order_flow", "hf_ohlcv_1m")
        self.store.write_jsonl(raw_path, raw_rows)
        self.store.write_jsonl(normalized_path, event_rows)
        if self.table_store:
            self.table_store.write_table(self.tables.normalized("order_flow", "hf_ohlcv_1m"), event_rows)

        raw_manifest = Manifest(
            manifest_id="hf-ohlcv-1m-raw",
            artifact_type="raw",
            paths=[raw_path],
            metadata={
                "source": "huggingface_ohlcv_1m",
                "dataset": "mito0o852/OHLCV-1m",
                "record_count": str(len(raw_rows)),
            },
        )
        raw_manifest_path = self.paths.manifest("ingest", "hf-ohlcv-1m-raw")
        self.store.write_manifest(raw_manifest_path, raw_manifest)

        normalization_manifest = Manifest(
            manifest_id="hf-ohlcv-1m-order-flow-normalized",
            artifact_type="normalized",
            paths=[normalized_path],
            metadata={
                "source": "huggingface_ohlcv_1m",
                "source_manifest_id": raw_manifest.manifest_id,
                "order_flow": str(len(event_rows)),
                "feature_contract": "paper-order-flow-v1",
            },
        )
        normalization_manifest_path = self.paths.manifest("normalization", "hf-ohlcv-1m-order-flow-normalized")
        self.store.write_manifest(normalization_manifest_path, normalization_manifest)

        return IngestResult(
            raw_manifest_path=raw_manifest_path,
            normalization_manifest_path=normalization_manifest_path,
            normalized_counts={"order_flow": len(event_rows)},
            quality_summary={"quarantined_records": 0, "duplicate_records": 0},
        )


def _ohlcv_rows_to_order_flow(rows: list[dict[str, object]]) -> list[OrderFlowEventRecord]:
    events: list[OrderFlowEventRecord] = []
    previous_by_ticker: dict[str, dict[str, object]] = {}
    previous_midprice_by_ticker: dict[str, float] = {}
    volume_history_by_ticker: dict[str, list[float]] = {}
    for row in rows:
        ticker = str(row["ticker"]).upper()
        timestamp = _timestamp_to_utc(row["timestamp"])
        midprice = _estimate_midprice(row)
        volume = float(row["volume"])
        previous = previous_by_ticker.get(ticker)
        if previous is None:
            previous_by_ticker[ticker] = row
            previous_midprice_by_ticker[ticker] = midprice
            volume_history_by_ticker[ticker] = [volume]
            continue
        previous_time = _timestamp_to_utc(previous["timestamp"])
        previous_midprice = max(previous_midprice_by_ticker[ticker], 1e-12)
        order_price = max(float(row["close"]), 1e-12)
        relative_price_bps = 10_000.0 * math.log(order_price / midprice)
        midprice_return_bps = 10_000.0 * math.log(midprice / previous_midprice)
        side = "buy" if relative_price_bps >= 0.0 else "sell"
        depth = abs(relative_price_bps)
        volume_baseline = median(volume_history_by_ticker[ticker])
        relative_size = volume / max(volume_baseline, 1.0)
        action = "add" if volume >= float(previous["volume"]) else "delete"
        event_id = f"hf-ohlcv-1m-{ticker}-{timestamp.isoformat().replace('+00:00', 'Z')}"
        events.append(
            OrderFlowEventRecord(
                event_id=event_id,
                ticker=ticker,
                timestamp=timestamp.isoformat().replace("+00:00", "Z"),
                date=timestamp.date().isoformat(),
                action=action,
                side=side,
                midprice=midprice,
                relative_price_bps=relative_price_bps,
                price_depth_bps=depth,
                size=relative_size,
                interarrival_seconds=max((timestamp - previous_time).total_seconds(), 1.0),
                provider="hf_ohlcv_1m",
                source_ids=[event_id],
                midprice_return_bps=midprice_return_bps,
            )
        )
        previous_by_ticker[ticker] = row
        previous_midprice_by_ticker[ticker] = midprice
        volume_history_by_ticker[ticker].append(volume)
    return events


def _estimate_midprice(row: dict[str, object]) -> float:
    high = _optional_float(row.get("high"))
    low = _optional_float(row.get("low"))
    close = _optional_float(row.get("close"))
    if high is not None and low is not None and high > 0.0 and low > 0.0 and high >= low:
        return (high + low) / 2.0
    if close is not None and close > 0.0:
        return close
    raise ValueError(f"cannot estimate midprice for row: {row}")


def _fetch_parquet_table(url: str) -> pa.Table:
    request = Request(url, headers={"User-Agent": "Mega-Trading"})
    context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(request, timeout=120, context=context) as response:
        data = response.read()
    with NamedTemporaryFile(suffix=".parquet") as tmp:
        tmp.write(data)
        tmp.flush()
        return pq.read_table(tmp.name, columns=["timestamp", "open", "high", "low", "close", "volume", "ticker"])


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _to_utc_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _timestamp_to_utc(value: object) -> datetime:
    if isinstance(value, datetime):
        timestamp = value
    else:
        timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _month_range(start: datetime, end: datetime) -> list[str]:
    month = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    last = datetime(end.year, end.month, 1, tzinfo=timezone.utc)
    months: list[str] = []
    while month <= last:
        months.append(f"{month.year:04d}-{month.month:02d}")
        year = month.year + (1 if month.month == 12 else 0)
        next_month = 1 if month.month == 12 else month.month + 1
        month = datetime(year, next_month, 1, tzinfo=timezone.utc)
    return months
