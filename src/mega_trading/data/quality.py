"""Data quality gates and readiness reports for normalized records."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from mega_trading.core.schemas import Manifest
from mega_trading.core.store import LocalObjectStore


@dataclass(frozen=True)
class DataQualityResult:
    passed: bool
    report_path: str
    manifest_path: str
    metrics_path: str
    quarantine_path: str
    total_records: int
    quarantined_records: int


class DataQualityChecker:
    def __init__(self, store: LocalObjectStore) -> None:
        self.store = store

    def run(self, normalization_manifest_paths: list[str], run_id: str = "latest", fail_on_error: bool = False) -> DataQualityResult:
        normalized_paths = self._normalized_paths(normalization_manifest_paths)
        quarantine_rows: list[dict[str, Any]] = []
        total_records = 0

        for path in normalized_paths:
            seen_ids: set[str] = set()
            for index, row in enumerate(self.store.read_jsonl(path)):
                total_records += 1
                reasons = self._reasons(path, row)
                record_id = _record_id(path, row)
                if record_id:
                    if record_id in seen_ids:
                        reasons.append("duplicate_id")
                    seen_ids.add(record_id)
                for reason in reasons:
                    quarantine_rows.append(
                        {
                            "path": path,
                            "row_index": index,
                            "record_id": record_id or "",
                            "reason": reason,
                        }
                    )

        issues_by_reason = dict(sorted(Counter(row["reason"] for row in quarantine_rows).items()))
        passed = not quarantine_rows
        quality_score = 1.0 if total_records == 0 else max(0.0, 1.0 - (len(quarantine_rows) / total_records))

        quarantine_path = f"quarantine/quality/{run_id}.jsonl"
        metrics_path = "metrics/data/quality.jsonl"
        report_path = "reports/data-readiness.json"
        manifest_path = f"manifests/quality/{run_id}.json"

        self.store.write_jsonl(quarantine_path, quarantine_rows)
        self.store.write_jsonl(
            metrics_path,
            [
                {
                    "run_id": run_id,
                    "total_records": total_records,
                    "quarantined_records": len(quarantine_rows),
                    "quality_score": quality_score,
                    "passed": passed,
                }
            ],
        )
        self.store.write_json(
            report_path,
            {
                "run_id": run_id,
                "passed": passed,
                "training_ready": passed,
                "checked_paths": normalized_paths,
                "total_records": total_records,
                "quarantined_records": len(quarantine_rows),
                "quality_score": quality_score,
                "issues_by_reason": issues_by_reason,
            },
        )
        self.store.write_manifest(
            manifest_path,
            Manifest(
                manifest_id=f"{run_id}-quality",
                artifact_type="quality",
                paths=[report_path, metrics_path, quarantine_path],
                metadata={
                    "passed": str(passed).lower(),
                    "total_records": str(total_records),
                    "quarantined_records": str(len(quarantine_rows)),
                },
            ),
        )
        if fail_on_error and not passed:
            raise DataQualityError(f"data quality failed: {issues_by_reason}")
        return DataQualityResult(
            passed=passed,
            report_path=report_path,
            manifest_path=manifest_path,
            metrics_path=metrics_path,
            quarantine_path=quarantine_path,
            total_records=total_records,
            quarantined_records=len(quarantine_rows),
        )

    def _normalized_paths(self, manifest_paths: list[str]) -> list[str]:
        paths: list[str] = []
        for manifest_path in manifest_paths:
            manifest = self.store.read_manifest(manifest_path)
            paths.extend(path for path in manifest.paths if path.startswith("stage=02_normalized/") and path.endswith(".jsonl"))
        return paths

    def _reasons(self, path: str, row: dict[str, Any]) -> list[str]:
        reasons: list[str] = []
        for field in _required_fields(path):
            if row.get(field) in (None, "", []):
                reasons.append(f"missing_{field}")
        if "family=market_data" in path and float(row.get("adjusted_close") or 0.0) <= 0.0:
            reasons.append("invalid_market_data")
        if "date" in row and not _valid_date(str(row["date"])):
            reasons.append("invalid_date")
        for field in ("as_of_time", "accepted_at", "timestamp", "published_at"):
            if field in row and row[field] and not _valid_datetime(str(row[field])):
                reasons.append(f"invalid_{field}")
        if row.get("timestamp") and row.get("as_of_time") and _valid_datetime(str(row["timestamp"])) and _valid_datetime(str(row["as_of_time"])):
            if _parse_datetime(str(row["timestamp"])) > _parse_datetime(str(row["as_of_time"])):
                reasons.append("future_leakage")
        return reasons


class DataQualityError(ValueError):
    """Raised when quality checks fail and fail_on_error is enabled."""


def _required_fields(path: str) -> tuple[str, ...]:
    if "family=entities" in path:
        return ("entity_id", "ticker", "company_name")
    if "family=sec_filings" in path:
        return ("sec_filing_id", "entity_id", "ticker", "concept", "period_end", "accepted_at", "as_of_time", "source_ids")
    if "family=market_data" in path:
        return ("market_data_id", "ticker", "date", "adjusted_close", "provider", "source_ids")
    if "family=documents" in path:
        return ("document_id", "entity_id", "ticker", "source_type", "text", "source_uri", "as_of_time", "source_ids")
    return ()


def _record_id(path: str, row: dict[str, Any]) -> str | None:
    for field in _id_fields(path):
        value = row.get(field)
        if value:
            return str(value)
    return None


def _id_fields(path: str) -> tuple[str, ...]:
    if "family=entities" in path:
        return ("entity_id",)
    if "family=sec_filings" in path:
        return ("sec_filing_id",)
    if "family=market_data" in path:
        return ("market_data_id",)
    if "family=documents" in path:
        return ("document_id",)
    return ("entity_id", "sec_filing_id", "market_data_id", "document_id")


def _valid_date(value: str) -> bool:
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def _valid_datetime(value: str) -> bool:
    try:
        _parse_datetime(value)
        return True
    except ValueError:
        return False


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
