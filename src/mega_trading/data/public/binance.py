"""Binance public trade data adapter for event-level order-flow proxies."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
import io
import math
import ssl
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import Callable
from urllib.request import Request, urlopen
from zipfile import ZipFile

import certifi

from mega_trading.core.schemas import OrderFlowEventRecord
from mega_trading.data.ingest import BinanceTradesIngestRequest

FetchZip = Callable[[str], bytes]
BINANCE_TRADES_BASE_URL = "https://data.binance.vision/data/spot/monthly/trades"


def _load_binance_trade_rows(request: BinanceTradesIngestRequest, base_url: str, fetch_zip: FetchZip) -> list[dict[str, object]]:
    start_dt = _to_utc_datetime(request.start)
    end_dt = _to_utc_datetime(request.end)
    rows: list[dict[str, object]] = []
    tasks = _trade_file_tasks(request, base_url, start_dt, end_dt)
    with ThreadPoolExecutor(max_workers=request.download_workers) as executor:
        for chunk in executor.map(lambda task: _load_trade_file(task, fetch_zip, start_dt, end_dt), tasks):
            rows.extend(chunk)
    return sorted(rows, key=lambda row: (str(row["symbol"]), str(row["timestamp"]), str(row["trade_id"])))


def _trade_file_tasks(
    request: BinanceTradesIngestRequest,
    base_url: str,
    start_dt: datetime,
    end_dt: datetime,
) -> list[tuple[str, str]]:
    root = base_url.replace("/monthly/", f"/{request.frequency}/")
    tasks: list[tuple[str, str]] = []
    for symbol in sorted({symbol.upper() for symbol in request.symbols}):
        for partition in _date_partitions(start_dt, end_dt, request.frequency):
            tasks.append((symbol, f"{root}/{symbol}/{symbol}-trades-{partition}.zip"))
    return tasks


def _load_trade_file(
    task: tuple[str, str],
    fetch_zip: FetchZip,
    start_dt: datetime,
    end_dt: datetime,
) -> list[dict[str, object]]:
    symbol, url = task
    rows: list[dict[str, object]] = []
    for row in _parse_trades_zip(fetch_zip(url)):
        timestamp = _binance_timestamp_to_utc(int(row["time"]))
        if timestamp < start_dt or timestamp > end_dt:
            continue
        qty = float(row["qty"])
        price = float(row["price"])
        if qty <= 0.0 or price <= 0.0:
            continue
        rows.append(
            {
                "symbol": symbol,
                "trade_id": str(row["trade_id"]),
                "price": price,
                "qty": qty,
                "quote_qty": float(row["quote_qty"]),
                "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
                "is_buyer_maker": _parse_bool(row["is_buyer_maker"]),
                "is_best_match": _parse_bool(row.get("is_best_match", "true")),
            }
        )
    return rows


def _parse_trades_zip(payload: bytes) -> list[dict[str, str]]:
    with ZipFile(io.BytesIO(payload)) as archive:
        csv_name = next(name for name in archive.namelist() if name.endswith(".csv"))
        with archive.open(csv_name) as handle:
            text = io.TextIOWrapper(handle, encoding="utf-8")
            rows = []
            for fields in csv.reader(text):
                if not fields or fields[0] == "trade_id":
                    continue
                rows.append(
                    {
                        "trade_id": fields[0],
                        "price": fields[1],
                        "qty": fields[2],
                        "quote_qty": fields[3],
                        "time": fields[4],
                        "is_buyer_maker": fields[5],
                        "is_best_match": fields[6] if len(fields) > 6 else "true",
                    }
                )
            return rows


def _trade_rows_to_order_flow(rows: list[dict[str, object]]) -> list[OrderFlowEventRecord]:
    events: list[OrderFlowEventRecord] = []
    previous_by_symbol: dict[str, dict[str, object]] = {}
    qty_baseline_by_symbol = _quantity_baselines(rows)
    for row in rows:
        symbol = str(row["symbol"]).upper()
        timestamp = _to_utc_datetime(str(row["timestamp"]))
        price = float(row["price"])
        qty = float(row["qty"])
        previous = previous_by_symbol.get(symbol)
        if previous is None:
            previous_by_symbol[symbol] = row
            continue
        previous_price = max(float(previous["price"]), 1e-12)
        previous_time = _to_utc_datetime(str(previous["timestamp"]))
        relative_price_bps = 10_000.0 * math.log(price / previous_price)
        qty_baseline = qty_baseline_by_symbol[symbol]
        relative_size = qty / max(qty_baseline, 1e-12)
        side = "sell" if bool(row["is_buyer_maker"]) else "buy"
        event_id = f"binance-trades-{symbol}-{row['trade_id']}"
        events.append(
            OrderFlowEventRecord(
                event_id=event_id,
                ticker=symbol,
                timestamp=timestamp.isoformat().replace("+00:00", "Z"),
                date=timestamp.date().isoformat(),
                action="delete",
                side=side,
                midprice=price,
                relative_price_bps=relative_price_bps,
                price_depth_bps=abs(relative_price_bps),
                size=relative_size,
                interarrival_seconds=max((timestamp - previous_time).total_seconds(), 1e-6),
                provider="binance_trades",
                source_ids=[event_id],
                midprice_return_bps=relative_price_bps,
            )
        )
        previous_by_symbol[symbol] = row
    return events


def _quantity_baselines(rows: list[dict[str, object]]) -> dict[str, float]:
    qty_by_symbol: dict[str, list[float]] = {}
    for row in rows:
        qty_by_symbol.setdefault(str(row["symbol"]).upper(), []).append(float(row["qty"]))
    return {symbol: median(qtys) for symbol, qtys in qty_by_symbol.items()}


def _fetch_zip(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "Mega-Trading"})
    context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(request, timeout=120, context=context) as response:
        return response.read()


def _parse_bool(value: object) -> bool:
    return str(value).lower() == "true"


def _to_utc_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _binance_timestamp_to_utc(value: int) -> datetime:
    divisor = 1_000_000 if value >= 10**15 else 1_000
    return datetime.fromtimestamp(value / divisor, tz=timezone.utc)


def _date_partitions(start: datetime, end: datetime, frequency: str) -> list[str]:
    if frequency == "daily":
        return _day_range(start, end)
    if frequency == "monthly":
        return _month_range(start, end)
    raise ValueError("frequency must be daily or monthly")


def _day_range(start: datetime, end: datetime) -> list[str]:
    day = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
    last = datetime(end.year, end.month, end.day, tzinfo=timezone.utc)
    days: list[str] = []
    while day <= last:
        days.append(day.date().isoformat())
        day = day + timedelta(days=1)
    return days


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
