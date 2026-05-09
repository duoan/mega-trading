"""Multi-stream sample construction for the market foundation model."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
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
    """Build point-in-time price/fundamental/text samples with future labels."""

    def __init__(
        self,
        store: LocalObjectStore,
        label_config: LabelConfig | None = None,
        max_fundamentals: int = 16,
        max_evidence: int = 8,
        num_workers: int = 0,
    ) -> None:
        self.store = store
        self.label_config = label_config or LabelConfig()
        self.max_fundamentals = max_fundamentals
        self.max_evidence = max_evidence
        self.num_workers = num_workers
        self.paths = ArtifactPaths()

    def build(self, mixture_name: str = "public", run_id: str = "latest") -> SampleBuildResult:
        readiness = self.store.read_json("reports/data-readiness.json")
        if not readiness.get("training_ready"):
            raise ValueError("data readiness report is not training_ready")

        entities = {str(row["ticker"]): row for row in self._read_first_available(("entities", ("sec", "fixture")))}
        fundamentals = _group_by_ticker(self._read_first_available(("fundamentals", ("sec", "fixture"))))
        prices = self._read_prices()
        prices_by_ticker = _group_by_ticker(prices, sort_field="date")
        evidence = _group_by_ticker(self._read_first_available(("evidence", ("fixture", "sec"))))
        tasks = [
            {
                "ticker": ticker,
                "entity": entities[ticker],
                "fundamentals": fundamentals.get(ticker, []),
                "prices": prices_by_ticker[ticker],
                "evidence": evidence.get(ticker, []),
                "mixture_name": mixture_name,
                "label_config": self.label_config,
                "max_fundamentals": self.max_fundamentals,
                "max_evidence": self.max_evidence,
            }
            for ticker in sorted(entities)
            if ticker in prices_by_ticker
        ]
        sample_path = self.paths.corpus(mixture_name, "samples")
        workers = _worker_count(self.num_workers, len(tasks))
        if workers > 1:
            print(f"sample build using {workers} workers for {len(tasks)} tickers")
        samples = self.store.write_jsonl_iter(sample_path, _sample_rows(tasks, workers))

        manifest = Manifest(
            manifest_id=f"{mixture_name}-{run_id}-samples",
            artifact_type="samples",
            paths=[sample_path],
            metadata={
                "mixture_name": mixture_name,
                "samples": str(samples),
                "input_window_observations": str(self.label_config.input_window_observations),
                "horizon_observations": str(self.label_config.horizon_observations),
                "workers": str(workers),
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

    def _read_prices(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for source in ("yahoo", "stooq", "fixture"):
            try:
                rows.extend(self.store.read_jsonl(self.paths.normalized("prices", source)))
            except ArtifactNotFoundError:
                continue
        return rows


def _sample_rows(tasks: list[dict[str, Any]], workers: int) -> Iterable[dict[str, Any]]:
    if workers <= 1:
        for task in tasks:
            yield from _samples_for_ticker(task)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for samples in executor.map(_samples_for_ticker, tasks, chunksize=1):
            yield from samples


def _samples_for_ticker(task: dict[str, Any]) -> list[dict[str, Any]]:
    ticker = str(task["ticker"])
    prices = list(task["prices"])
    price_index_by_date = {str(row["date"]): index for index, row in enumerate(prices)}
    labels = ForwardLabelGenerator(task["label_config"])._generate_for_ticker(ticker, prices)
    return [
        _sample_from_label(
            label,
            task["entity"],
            task["fundamentals"],
            prices,
            price_index_by_date,
            task["evidence"],
            str(task["mixture_name"]),
            int(task["max_fundamentals"]),
            int(task["max_evidence"]),
            task["label_config"],
        )
        for label in labels
    ]


def _sample_from_label(
    label: dict[str, Any],
    entity: dict[str, Any],
    fundamentals: list[dict[str, Any]],
    prices: list[dict[str, Any]],
    price_index_by_date: dict[str, int],
    evidence: list[dict[str, Any]],
    mixture_name: str,
    max_fundamentals: int,
    max_evidence: int,
    label_config: LabelConfig,
) -> dict[str, Any]:
    ticker = str(label["ticker"])
    as_of_time = str(label["as_of_time"])
    price_window = _price_window(prices, price_index_by_date, str(label["as_of_date"]), label_config.input_window_observations)
    visible_fundamentals = _visible_by_as_of(fundamentals, as_of_time, "as_of_time", limit=max_fundamentals)
    visible_evidence = _visible_by_as_of(evidence, as_of_time, "as_of_time", limit=max_evidence)
    source_ids = _source_ids(price_window) + _source_ids(visible_fundamentals) + _source_ids(visible_evidence)
    evidence_ids = [str(row["evidence_id"]) for row in visible_evidence if row.get("evidence_id")]
    return {
        "sample_id": f"sample-{ticker}-{label['as_of_date']}",
        "mixture_name": mixture_name,
        "ticker": ticker,
        "entity_id": str(entity["entity_id"]),
        "as_of_time": as_of_time,
        "price_window": price_window,
        "fundamental_facts": visible_fundamentals,
        "text_evidence": visible_evidence,
        "labels": label,
        "source_ids": source_ids,
        "evidence_ids": evidence_ids,
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


def _price_window(prices: list[dict[str, Any]], index_by_date: dict[str, int], as_of_date: str, window: int) -> list[dict[str, Any]]:
    as_of_index = index_by_date[as_of_date]
    return prices[as_of_index - window + 1 : as_of_index + 1]


def _visible_by_as_of(rows: list[dict[str, Any]], as_of_time: str, time_field: str, limit: int) -> list[dict[str, Any]]:
    visible = [row for row in rows if str(row.get(time_field, "")) <= as_of_time]
    return sorted(visible, key=lambda row: str(row.get(time_field, "")), reverse=True)[:limit]


def _source_ids(rows: list[dict[str, Any]]) -> list[str]:
    source_ids: list[str] = []
    for row in rows:
        source_ids.extend(str(source_id) for source_id in row.get("source_ids", []))
        for id_field in ("price_id", "fundamental_id", "evidence_id"):
            if row.get(id_field):
                source_ids.append(str(row[id_field]))
    return sorted(set(source_ids))


def _worker_count(requested_workers: int, task_count: int) -> int:
    if task_count <= 1:
        return 1
    if requested_workers == 0:
        requested_workers = os.cpu_count() or 1
    return max(1, min(requested_workers, task_count))
