"""Multi-stream sample construction for the market foundation model."""

from __future__ import annotations

import os
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from mega_trading.core.schemas import Manifest
from mega_trading.core.store import ArtifactNotFoundError, ArtifactPaths, LocalObjectStore
from mega_trading.data.labels import ForwardLabelGenerator, LabelConfig


@dataclass(frozen=True)
class SampleBuildResult:
    sample_path: str
    manifest_path: str
    samples: int


class MultiStreamSampleBuilder:
    """Build point-in-time market_data/news/sec_filings/earnings/macro samples with future labels."""

    def __init__(
        self,
        store: LocalObjectStore,
        label_config: LabelConfig | None = None,
        max_news: int = 32,
        max_sec_filings: int = 16,
        max_earnings: int = 16,
        max_macro: int = 16,
        num_workers: int = 0,
    ) -> None:
        self.store = store
        self.label_config = label_config or LabelConfig()
        self.max_news = max_news
        self.max_sec_filings = max_sec_filings
        self.max_earnings = max_earnings
        self.max_macro = max_macro
        self.num_workers = num_workers
        self.paths = ArtifactPaths()

    def build(self, mixture_name: str = "public", run_id: str = "latest") -> SampleBuildResult:
        readiness = self.store.read_json("reports/data-readiness.json")
        if not readiness.get("training_ready"):
            raise ValueError("data readiness report is not training_ready")

        entities = {str(row["ticker"]): row for row in self._read_first_available(("entities", ("sec", "fixture")))}
        news = _group_by_ticker(self._read_all_available("news", ("fixture", "rss", "gdelt", "vendor")))
        sec_filings = _group_by_ticker(self._read_first_available(("sec_filings", ("sec", "fixture"))))
        earnings = _group_by_ticker(self._read_all_available("earnings", ("fixture", "sec", "vendor")))
        macro = self._read_all_available("macro", ("fixture", "fred", "market_proxy"))
        market_data = self._read_market_data()
        market_data_by_ticker = _group_by_ticker(market_data, sort_field="date")
        tasks = [
            {
                "ticker": ticker,
                "entity": entities[ticker],
                "news": news.get(ticker, []),
                "sec_filings": sec_filings.get(ticker, []),
                "earnings": earnings.get(ticker, []),
                "macro": macro,
                "market_data": market_data_by_ticker[ticker],
                "mixture_name": mixture_name,
                "label_config": self.label_config,
                "max_news": self.max_news,
                "max_sec_filings": self.max_sec_filings,
                "max_earnings": self.max_earnings,
                "max_macro": self.max_macro,
            }
            for ticker in sorted(entities)
            if ticker in market_data_by_ticker
        ]
        workers = _worker_count(self.num_workers, len(tasks))
        if workers > 1:
            print(f"sample build using {workers} workers for {len(tasks)} tickers")
            partition_root = f"stage=04_corpus/mixture={mixture_name}/partitions"
            partition_tasks = [
                {
                    **task,
                    "store_root": str(self.store.root),
                    "sample_path": f"{partition_root}/ticker={_safe_partition_value(str(task['ticker']))}/samples.jsonl",
                }
                for task in tasks
            ]
            with ProcessPoolExecutor(max_workers=workers) as executor:
                partition_results = list(executor.map(_write_ticker_sample_partition, partition_tasks, chunksize=1))
            sample_paths = [path for path, _count in partition_results]
            samples = sum(count for _path, count in partition_results)
            sample_path = partition_root
        else:
            sample_path = self.paths.corpus(mixture_name, "samples")
            samples = self.store.write_jsonl_iter(sample_path, _sample_rows(tasks))
            sample_paths = [sample_path]

        manifest = Manifest(
            manifest_id=f"{mixture_name}-{run_id}-samples",
            artifact_type="samples",
            paths=sample_paths,
            metadata={
                "mixture_name": mixture_name,
                "samples": str(samples),
                "input_window_observations": str(self.label_config.input_window_observations),
                "horizon_observations": str(self.label_config.horizon_observations),
                "workers": str(workers),
                "partitioned": str(workers > 1).lower(),
                "readiness_quality_score": str(readiness.get("quality_score", "")),
            },
        )
        manifest_path = self.paths.manifest("samples", f"{mixture_name}-{run_id}-samples")
        self.store.write_manifest(manifest_path, manifest)
        return SampleBuildResult(sample_path=sample_path, manifest_path=manifest_path, samples=samples)

    def _read_first_available(self, family_and_sources: tuple[str, tuple[str, ...]]) -> list[dict[str, Any]]:
        family, sources = family_and_sources
        for source in sources:
            path = self.paths.normalized(family, source)
            try:
                rows = self.store.read_jsonl(path)
            except ArtifactNotFoundError:
                continue
            if rows:
                return rows
        return []

    def _read_all_available(self, family: str, sources: tuple[str, ...]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for source in sources:
            path = self.paths.normalized(family, source)
            try:
                rows.extend(self.store.read_jsonl(path))
            except ArtifactNotFoundError:
                continue
        return rows

    def _read_market_data(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for source in ("yahoo", "stooq", "fixture"):
            try:
                rows.extend(self.store.read_jsonl(self.paths.normalized("market_data", source)))
            except ArtifactNotFoundError:
                continue
        return rows


def _sample_rows(tasks: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for task in tasks:
        yield from _samples_for_ticker(task)


def _write_ticker_sample_partition(task: dict[str, Any]) -> tuple[str, int]:
    sample_path = str(task["sample_path"])
    target = Path(str(task["store_root"])) / sample_path
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for sample in _samples_for_ticker(task):
            handle.write(json.dumps(sample, sort_keys=True) + "\n")
            count += 1
    return sample_path, count


def _samples_for_ticker(task: dict[str, Any]) -> list[dict[str, Any]]:
    ticker = str(task["ticker"])
    market_data = list(task["market_data"])
    market_data_index_by_date = {str(row["date"]): index for index, row in enumerate(market_data)}
    labels = ForwardLabelGenerator(task["label_config"])._generate_for_ticker(ticker, market_data)
    return [
        _sample_from_label(
            label,
            task["entity"],
            task["news"],
            task["sec_filings"],
            task["earnings"],
            task["macro"],
            market_data,
            market_data_index_by_date,
            str(task["mixture_name"]),
            int(task["max_news"]),
            int(task["max_sec_filings"]),
            int(task["max_earnings"]),
            int(task["max_macro"]),
            task["label_config"],
        )
        for label in labels
    ]


def _sample_from_label(
    label: dict[str, Any],
    entity: dict[str, Any],
    news: list[dict[str, Any]],
    sec_filings: list[dict[str, Any]],
    earnings: list[dict[str, Any]],
    macro: list[dict[str, Any]],
    market_data: list[dict[str, Any]],
    market_data_index_by_date: dict[str, int],
    mixture_name: str,
    max_news: int,
    max_sec_filings: int,
    max_earnings: int,
    max_macro: int,
    label_config: LabelConfig,
) -> dict[str, Any]:
    ticker = str(label["ticker"])
    as_of_time = str(label["as_of_time"])
    market_data_window = _market_data_window(market_data, market_data_index_by_date, str(label["as_of_date"]), label_config.input_window_observations)
    visible_news = _visible_by_as_of(news, as_of_time, limit=max_news)
    visible_sec_filings = _visible_by_as_of(sec_filings, as_of_time, "as_of_time", limit=max_sec_filings)
    visible_earnings = _visible_by_as_of(earnings, as_of_time, "as_of_time", limit=max_earnings)
    visible_macro = _visible_by_as_of(macro, as_of_time, limit=max_macro)
    source_ids = (
        _source_ids(market_data_window)
        + _source_ids(visible_news)
        + _source_ids(visible_sec_filings)
        + _source_ids(visible_earnings)
        + _source_ids(visible_macro)
    )
    return {
        "sample_id": f"sample-{ticker}-{label['as_of_date']}",
        "mixture_name": mixture_name,
        "ticker": ticker,
        "entity_id": str(entity["entity_id"]),
        "as_of_time": as_of_time,
        "market_data_window": market_data_window,
        "news_window": visible_news,
        "sec_filing_window": visible_sec_filings,
        "earnings_window": visible_earnings,
        "macro_window": visible_macro,
        "labels": label,
        "source_ids": source_ids,
        "metadata": {
            "input_window_observations": label_config.input_window_observations,
            "horizon_observations": label_config.horizon_observations,
        },
    }


def _group_by_ticker(rows: list[dict[str, Any]], sort_field: str | None = None) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        ticker = str(row.get("ticker", ""))
        if ticker:
            grouped.setdefault(ticker, []).append(row)
    if sort_field:
        return {ticker: sorted(items, key=lambda row: str(row[sort_field])) for ticker, items in grouped.items()}
    return grouped


def _market_data_window(market_data: list[dict[str, Any]], index_by_date: dict[str, int], as_of_date: str, window: int) -> list[dict[str, Any]]:
    as_of_index = index_by_date[as_of_date]
    return market_data[as_of_index - window + 1 : as_of_index + 1]


def _visible_by_as_of(rows: list[dict[str, Any]], as_of_time: str, time_field: str = "timestamp", limit: int = 16) -> list[dict[str, Any]]:
    visible = [row for row in rows if _event_time(row, time_field) <= as_of_time]
    return sorted(visible, key=lambda row: _event_time(row, time_field), reverse=True)[:limit]


def _event_time(row: dict[str, Any], preferred_field: str) -> str:
    for field in (preferred_field, "as_of_time", "timestamp", "accepted_at", "date"):
        value = row.get(field)
        if value:
            return str(value)
    return ""


def _source_ids(rows: list[dict[str, Any]]) -> list[str]:
    source_ids: list[str] = []
    for row in rows:
        source_ids.extend(str(source_id) for source_id in row.get("source_ids", []))
        for id_field in ("market_data_id", "event_id", "sec_filing_id", "earnings_id", "macro_id"):
            if row.get(id_field):
                source_ids.append(str(row[id_field]))
    return sorted(set(source_ids))


def _worker_count(requested_workers: int, task_count: int) -> int:
    if task_count <= 1:
        return 1
    if requested_workers == 0:
        requested_workers = os.cpu_count() or 1
    return max(1, min(requested_workers, task_count))


def _safe_partition_value(value: str) -> str:
    return value.replace("/", "_")
