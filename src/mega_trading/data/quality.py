"""Data quality gates for normalized order-flow records."""

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
                reasons = self._reasons(row)
                record_id = str(row.get("event_id") or "")
                if record_id:
                    if record_id in seen_ids:
                        reasons.append("duplicate_id")
                    seen_ids.add(record_id)
                for reason in reasons:
                    quarantine_rows.append(
                        {
                            "path": path,
                            "row_index": index,
                            "record_id": record_id,
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
            paths.extend(
                path
                for path in manifest.paths
                if path.startswith("stage=02_normalized/family=order_flow/") and path.endswith(".jsonl")
            )
        return paths

    def _reasons(self, row: dict[str, Any]) -> list[str]:
        reasons: list[str] = []
        for field in (
            "event_id",
            "ticker",
            "timestamp",
            "date",
            "action",
            "side",
            "midprice",
            "relative_price_bps",
            "price_depth_bps",
            "size",
            "interarrival_seconds",
            "provider",
            "source_ids",
        ):
            if row.get(field) in (None, "", []):
                reasons.append(f"missing_{field}")
        if row.get("action") not in {"add", "delete"}:
            reasons.append("invalid_action")
        if row.get("side") not in {"buy", "sell"}:
            reasons.append("invalid_side")
        if float(row.get("midprice") or 0.0) <= 0.0:
            reasons.append("invalid_midprice")
        if float(row.get("price_depth_bps") or -1.0) < 0.0:
            reasons.append("invalid_price_depth")
        if float(row.get("size") or 0.0) <= 0.0:
            reasons.append("invalid_size")
        if float(row.get("interarrival_seconds") or 0.0) <= 0.0:
            reasons.append("invalid_interarrival_seconds")
        if row.get("date") and not _valid_date(str(row["date"])):
            reasons.append("invalid_date")
        if row.get("timestamp") and not _valid_datetime(str(row["timestamp"])):
            reasons.append("invalid_timestamp")
        return reasons


class DataQualityError(ValueError):
    """Raised when quality checks fail and fail_on_error is enabled."""


def _valid_date(value: str) -> bool:
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def _valid_datetime(value: str) -> bool:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False
