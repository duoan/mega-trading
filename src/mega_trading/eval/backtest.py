"""Deterministic backtest reports over labeled prediction logs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math

from mega_trading.core.schemas import BacktestResultRecord, Manifest
from mega_trading.core.store import ArtifactPaths, LocalObjectStore


@dataclass(frozen=True)
class BacktestConfig:
    fee_bps: float = 1.0
    slippage_bps: float = 1.0
    confidence_threshold: float = 0.0


@dataclass(frozen=True)
class BacktestResult:
    report_path: str
    manifest_path: str
    metrics: dict[str, float]


def run_backtest(
    store: LocalObjectStore,
    *,
    backtest_id: str,
    labeled_prediction_path: str,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    config = config or BacktestConfig()
    rows = [row for row in store.iter_jsonl(labeled_prediction_path) if row.get("label_status") == "ready"]
    metrics = _metrics(rows, config)
    paths = ArtifactPaths()
    report_path = paths.eval(backtest_id, "backtest-report")
    record = BacktestResultRecord(
        backtest_id=backtest_id,
        created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        prediction_path=str(rows[0].get("metadata", {}).get("prediction_path") or labeled_prediction_path)
        if rows
        else labeled_prediction_path,
        labeled_prediction_path=labeled_prediction_path,
        metrics=metrics,
        config={
            "fee_bps": config.fee_bps,
            "slippage_bps": config.slippage_bps,
            "confidence_threshold": config.confidence_threshold,
        },
        source_ids=[labeled_prediction_path],
    )
    store.write_json(report_path, record.to_dict())
    manifest = Manifest(
        manifest_id=f"{backtest_id}-backtest-report",
        artifact_type="backtest_report",
        paths=[report_path],
        metadata={
            "backtest_id": backtest_id,
            "labeled_prediction_path": labeled_prediction_path,
            "prediction_count": str(len(rows)),
        },
    )
    manifest_path = paths.manifest("evals", f"{backtest_id}-backtest-report")
    store.write_manifest(manifest_path, manifest)
    return BacktestResult(report_path=report_path, manifest_path=manifest_path, metrics=metrics)


def _metrics(rows: list[dict[str, object]], config: BacktestConfig) -> dict[str, float]:
    if not rows:
        return {
            "prediction_count": 0.0,
            "return_accuracy": 0.0,
            "risk_accuracy": 0.0,
            "actionable_signals": 0.0,
            "hit_rate": 0.0,
            "mean_slippage_adjusted_return": 0.0,
            "turnover": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
        }
    returns: list[float] = []
    signals: list[int] = []
    return_correct = 0
    risk_correct = 0
    cost = (config.fee_bps + config.slippage_bps) / 10_000.0
    for row in rows:
        return_correct += int(row.get("pred_return_bucket") == row.get("actual_return_bucket"))
        risk_correct += int(row.get("pred_risk_bucket") == row.get("actual_risk_bucket"))
        signal = _signal(row, config.confidence_threshold)
        signals.append(signal)
        if signal == 0:
            returns.append(0.0)
            continue
        returns.append(signal * float(row.get("actual_forward_return", 0.0)) - cost)
    actionable = sum(1 for signal in signals if signal != 0)
    hits = sum(1 for value in returns if value > 0)
    return {
        "prediction_count": float(len(rows)),
        "return_accuracy": return_correct / len(rows),
        "risk_accuracy": risk_correct / len(rows),
        "actionable_signals": float(actionable),
        "hit_rate": hits / actionable if actionable else 0.0,
        "mean_slippage_adjusted_return": sum(returns) / len(returns),
        "turnover": _turnover(signals),
        "max_drawdown": _max_drawdown(returns),
        "sharpe": _sharpe(returns),
    }


def _signal(row: dict[str, object], confidence_threshold: float) -> int:
    if float(row.get("confidence", 0.0)) < confidence_threshold:
        return 0
    if row.get("pred_return_bucket") == "outperform":
        return 1
    if row.get("pred_return_bucket") == "underperform":
        return -1
    return 0


def _turnover(signals: list[int]) -> float:
    if len(signals) < 2:
        return 0.0
    changes = sum(1 for index in range(1, len(signals)) if signals[index] != signals[index - 1])
    return changes / (len(signals) - 1)


def _max_drawdown(returns: list[float]) -> float:
    equity = 1.0
    peak = 1.0
    drawdown = 0.0
    for value in returns:
        equity *= 1.0 + value
        peak = max(peak, equity)
        drawdown = min(drawdown, (equity / peak) - 1.0)
    return drawdown


def _sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
    if variance == 0.0:
        return 0.0
    return mean / math.sqrt(variance) * math.sqrt(len(returns))
