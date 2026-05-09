"""Public OHLCV-to-event builder."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from statistics import median
from typing import Any, Iterable

from mega_trading.config import BuildConfig
from mega_trading.core.schemas import Manifest
from mega_trading.core.store import LocalObjectStore
from mega_trading.tokenizer import BOS_TOKEN, EOS_TOKEN, MarketEventTokenizer

STREAM_CONTRACT = "market-event-token-v1"


@dataclass(frozen=True)
class BuildResult:
    event_path: str
    shard_path: str
    profile_path: str
    manifest_path: str


@dataclass(frozen=True)
class _MarketPoint:
    ticker: str
    day: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    source_id: str


class EventBuilder:
    """Build event-like token streams from public daily market data."""

    def __init__(self, store: LocalObjectStore, config: BuildConfig | None = None) -> None:
        self.store = store
        self.config = config or BuildConfig()
        self.tokenizer = MarketEventTokenizer()

    def build(self) -> BuildResult:
        points_by_ticker = _market_points_by_ticker(self.store, self.config.source)
        if self.config.max_tickers is not None:
            selected = sorted(points_by_ticker)[: self.config.max_tickers]
            points_by_ticker = {ticker: points_by_ticker[ticker] for ticker in selected}

        events_by_ticker: dict[str, list[dict[str, Any]]] = {}
        for ticker, points in sorted(points_by_ticker.items()):
            events = _events_for_ticker(ticker, points)
            if len(events) >= self.config.min_events_per_ticker:
                events_by_ticker[ticker] = events
        if not events_by_ticker:
            raise ValueError("no tickers had enough market data to build events")

        event_path = f"stage=04_corpus/mixture={self.config.mixture_name}/events.jsonl"
        shard_path = f"stage=05_shards/mixture={self.config.mixture_name}/tokens.jsonl"
        profile_path = f"stage=05_shards/mixture={self.config.mixture_name}/tokens-profile.json"
        manifest_path = f"manifests/build/{self.config.mixture_name}.json"

        event_rows = [event for ticker in sorted(events_by_ticker) for event in events_by_ticker[ticker]]
        self.store.write_jsonl(event_path, event_rows)

        sequence_rows = list(_sequence_rows(events_by_ticker, self.tokenizer, self.config.block_size, self.config.stride))
        if not sequence_rows:
            raise ValueError("not enough events to build token sequences")
        self.store.write_jsonl(shard_path, sequence_rows)

        profile = _profile(event_rows, sequence_rows, self.tokenizer, self.config)
        self.store.write_json(profile_path, profile)
        self.store.write_manifest(
            manifest_path,
            Manifest(
                manifest_id=f"{self.config.mixture_name}-build",
                artifact_type="event-token-build",
                paths=[event_path, shard_path, profile_path],
                metadata=profile,
            ),
        )
        return BuildResult(
            event_path=event_path,
            shard_path=shard_path,
            profile_path=profile_path,
            manifest_path=manifest_path,
        )


def _market_points_by_ticker(store: LocalObjectStore, source: str) -> dict[str, list[_MarketPoint]]:
    path = f"stage=02_normalized/family=market_data/source={source}.jsonl"
    rows = store.read_jsonl(path)
    points: dict[str, list[_MarketPoint]] = defaultdict(list)
    for row in rows:
        ticker = str(row["ticker"]).upper()
        close = _float(row.get("adjusted_close", row.get("close")))
        points[ticker].append(
            _MarketPoint(
                ticker=ticker,
                day=date.fromisoformat(str(row["date"])),
                open=_float(row.get("open", close)),
                high=_float(row.get("high", close)),
                low=_float(row.get("low", close)),
                close=close,
                volume=max(_float(row.get("volume", 0.0)), 0.0),
                source_id=str(row.get("market_data_id", row.get("source_ids", [f"{source}:{ticker}:{row['date']}"])[0])),
            )
        )
    return {ticker: sorted(values, key=lambda point: point.day) for ticker, values in points.items()}


def _events_for_ticker(ticker: str, points: list[_MarketPoint]) -> list[dict[str, Any]]:
    if len(points) < 2:
        return []
    positive_volumes = [point.volume for point in points if point.volume > 0]
    median_volume = median(positive_volumes) if positive_volumes else 1.0
    events: list[dict[str, Any]] = []
    previous = points[0]
    for index, point in enumerate(points[1:], start=1):
        if previous.close <= 0 or point.close <= 0:
            previous = point
            continue
        return_bps = 10_000.0 * math.log(point.close / previous.close)
        gap_bps = 10_000.0 * math.log(max(point.open, 1e-12) / previous.close)
        range_bps = 10_000.0 * max(point.high - point.low, 0.0) / previous.close
        log_volume_ratio = math.log(max(point.volume, 1.0) / max(median_volume, 1.0))
        events.append(
            {
                "event_id": f"{ticker}-{point.day.isoformat()}",
                "ticker": ticker,
                "event_index": index,
                "date": point.day.isoformat(),
                "side": _side(return_bps),
                "return_bps": return_bps,
                "range_bps": range_bps,
                "gap_bps": gap_bps,
                "log_volume_ratio": log_volume_ratio,
                "dt_days": max((point.day - previous.day).days, 1),
                "weekday": point.day.weekday(),
                "source_ids": [previous.source_id, point.source_id],
            }
        )
        previous = point
    return events


def _sequence_rows(
    events_by_ticker: dict[str, list[dict[str, Any]]],
    tokenizer: MarketEventTokenizer,
    block_size: int,
    stride: int,
) -> Iterable[dict[str, Any]]:
    sequence_id = 0
    target_length = block_size + 1
    per_ticker_rows: list[list[dict[str, Any]]] = []
    for ticker, events in sorted(events_by_ticker.items()):
        tokens = [BOS_TOKEN]
        dates = ["bos"]
        for event in events:
            tokens.extend(tokenizer.encode_event(event))
            dates.extend([str(event["date"])] * tokenizer.event_size)
        tokens.append(EOS_TOKEN)
        dates.append("eos")

        rows: list[dict[str, Any]] = []
        for start in range(0, max(len(tokens) - target_length + 1, 0), stride):
            chunk = tokens[start : start + target_length]
            if len(chunk) == target_length:
                rows.append(
                    {
                        "sequence_id": f"seq-{sequence_id:08d}",
                        "ticker": ticker,
                        "start_date": dates[start],
                        "end_date": dates[start + target_length - 1],
                        "tokens": chunk,
                    }
                )
                sequence_id += 1
        per_ticker_rows.append(rows)

    for rows in _round_robin(per_ticker_rows):
        yield rows


def _profile(
    events: list[dict[str, Any]],
    sequences: list[dict[str, Any]],
    tokenizer: MarketEventTokenizer,
    config: BuildConfig,
) -> dict[str, Any]:
    ticker_counts = Counter(str(event["ticker"]) for event in events)
    sequence_counts = Counter(str(row["ticker"]) for row in sequences)
    return {
        "stream_contract": STREAM_CONTRACT,
        "mixture": config.mixture_name,
        "source": config.source,
        "event_count": len(events),
        "sequence_count": len(sequences),
        "ticker_counts": dict(sorted(ticker_counts.items())),
        "sequence_counts": dict(sorted(sequence_counts.items())),
        "block_size": config.block_size,
        "stride": config.stride,
        "event_size": tokenizer.event_size,
        "vocab_size": tokenizer.vocab_size,
    }


def _side(return_bps: float) -> str:
    if return_bps > 5.0:
        return "buy"
    if return_bps < -5.0:
        return "sell"
    return "flat"


def _float(value: object) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def _round_robin(groups: list[list[dict[str, Any]]]) -> Iterable[dict[str, Any]]:
    max_len = max((len(group) for group in groups), default=0)
    for index in range(max_len):
        for group in groups:
            if index < len(group):
                yield group[index]
