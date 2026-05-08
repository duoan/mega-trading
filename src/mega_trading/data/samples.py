"""Multi-stream sample construction for the market foundation model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    ) -> None:
        self.store = store
        self.label_config = label_config or LabelConfig()
        self.max_fundamentals = max_fundamentals
        self.max_evidence = max_evidence
        self.paths = ArtifactPaths()

    def build(self, mixture_name: str = "public", run_id: str = "latest") -> SampleBuildResult:
        readiness = self.store.read_json("reports/data-readiness.json")
        if not readiness.get("training_ready"):
            raise ValueError("data readiness report is not training_ready")

        entities = {str(row["ticker"]): row for row in self._read_first_available(("entities", ("sec", "fixture")))}
        fundamentals = _group_by_ticker(self._read_first_available(("fundamentals", ("sec", "fixture"))))
        prices = self._read_prices()
        prices_by_ticker = _group_by_ticker(prices)
        evidence = _group_by_ticker(self._read_first_available(("evidence", ("fixture", "sec"))))
        labels = ForwardLabelGenerator(self.label_config).generate(prices)

        samples = [
            self._sample_from_label(label, entities, fundamentals, prices_by_ticker, evidence, mixture_name)
            for label in labels
            if str(label["ticker"]) in entities
        ]
        sample_path = self.paths.corpus(mixture_name, "samples")
        self.store.write_jsonl(sample_path, samples)

        manifest = Manifest(
            manifest_id=f"{mixture_name}-{run_id}-samples",
            artifact_type="samples",
            paths=[sample_path],
            metadata={
                "mixture_name": mixture_name,
                "samples": str(len(samples)),
                "input_window_observations": str(self.label_config.input_window_observations),
                "horizon_observations": str(self.label_config.horizon_observations),
                "readiness_quality_score": str(readiness.get("quality_score", "")),
            },
        )
        manifest_path = self.paths.manifest("samples", f"{mixture_name}-{run_id}-samples")
        self.store.write_manifest(manifest_path, manifest)
        return SampleBuildResult(sample_path=sample_path, manifest_path=manifest_path, samples=len(samples))

    def _sample_from_label(
        self,
        label: dict[str, Any],
        entities: dict[str, dict[str, Any]],
        fundamentals: dict[str, list[dict[str, Any]]],
        prices_by_ticker: dict[str, list[dict[str, Any]]],
        evidence: dict[str, list[dict[str, Any]]],
        mixture_name: str,
    ) -> dict[str, Any]:
        ticker = str(label["ticker"])
        as_of_time = str(label["as_of_time"])
        price_window = _price_window(
            prices_by_ticker[ticker],
            str(label["as_of_date"]),
            self.label_config.input_window_observations,
        )
        visible_fundamentals = _visible_by_as_of(
            fundamentals.get(ticker, []),
            as_of_time,
            "as_of_time",
            limit=self.max_fundamentals,
        )
        visible_evidence = _visible_by_as_of(
            evidence.get(ticker, []),
            as_of_time,
            "as_of_time",
            limit=self.max_evidence,
        )
        source_ids = _source_ids(price_window) + _source_ids(visible_fundamentals) + _source_ids(visible_evidence)
        evidence_ids = [str(row["evidence_id"]) for row in visible_evidence if row.get("evidence_id")]
        return {
            "sample_id": f"sample-{ticker}-{label['as_of_date']}",
            "mixture_name": mixture_name,
            "ticker": ticker,
            "entity_id": str(entities[ticker]["entity_id"]),
            "as_of_time": as_of_time,
            "price_window": price_window,
            "fundamental_facts": visible_fundamentals,
            "text_evidence": visible_evidence,
            "labels": label,
            "source_ids": source_ids,
            "evidence_ids": evidence_ids,
            "metadata": {
                "input_window_observations": self.label_config.input_window_observations,
                "horizon_observations": self.label_config.horizon_observations,
            },
        }

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


def _group_by_ticker(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        ticker = str(row.get("ticker", ""))
        if ticker:
            grouped.setdefault(ticker, []).append(row)
    return grouped


def _price_window(prices: list[dict[str, Any]], as_of_date: str, window: int) -> list[dict[str, Any]]:
    sorted_prices = sorted(prices, key=lambda row: str(row["date"]))
    as_of_index = next(index for index, row in enumerate(sorted_prices) if str(row["date"]) == as_of_date)
    return sorted_prices[as_of_index - window + 1 : as_of_index + 1]


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
