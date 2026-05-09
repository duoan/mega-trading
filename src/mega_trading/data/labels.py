"""Point-in-time label generation for market model samples."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class LabelConfig:
    input_window_observations: int = 20
    horizon_observations: int = 20
    return_threshold: float = 0.02
    high_risk_drawdown: float = -0.05
    medium_risk_drawdown: float = -0.02
    high_risk_volatility: float = 0.03
    medium_risk_volatility: float = 0.015

    def __post_init__(self) -> None:
        if self.input_window_observations < 1:
            raise ValueError("input_window_observations must be positive")
        if self.horizon_observations < 1:
            raise ValueError("horizon_observations must be positive")
        if self.return_threshold <= 0:
            raise ValueError("return_threshold must be positive")


class ForwardLabelGenerator:
    """Generate future labels without exposing future market_data as inputs."""

    def __init__(self, config: LabelConfig | None = None) -> None:
        self.config = config or LabelConfig()

    def generate(self, market_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        labels: list[dict[str, Any]] = []
        for ticker, ticker_market_data in _group_market_data(market_data).items():
            labels.extend(self._generate_for_ticker(ticker, ticker_market_data))
        return labels

    def _generate_for_ticker(self, ticker: str, market_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        labels: list[dict[str, Any]] = []
        stop_index = len(market_data) - self.config.horizon_observations
        for as_of_index in range(self.config.input_window_observations - 1, stop_index):
            as_of_market_data = market_data[as_of_index]
            label_start = market_data[as_of_index + 1]
            label_end = market_data[as_of_index + self.config.horizon_observations]
            start_close = float(as_of_market_data["adjusted_close"])
            end_close = float(label_end["adjusted_close"])
            if start_close <= 0 or end_close <= 0:
                continue
            forward_return = (end_close / start_close) - 1.0
            future_market_data = market_data[as_of_index : as_of_index + self.config.horizon_observations + 1]
            max_drawdown = _max_drawdown([float(row["adjusted_close"]) for row in future_market_data])
            realized_volatility = _realized_volatility([float(row["adjusted_close"]) for row in future_market_data])
            labels.append(
                {
                    "label_id": f"label-{ticker}-{as_of_market_data['date']}-{self.config.horizon_observations}d",
                    "ticker": ticker,
                    "as_of_date": str(as_of_market_data["date"]),
                    "as_of_time": f"{as_of_market_data['date']}T00:00:00Z",
                    "label_start_date": str(label_start["date"]),
                    "label_end_date": str(label_end["date"]),
                    "horizon_observations": self.config.horizon_observations,
                    "start_market_data_id": str(as_of_market_data["market_data_id"]),
                    "end_market_data_id": str(label_end["market_data_id"]),
                    "forward_return": forward_return,
                    "forward_return_bucket": _return_bucket(forward_return, self.config.return_threshold),
                    "max_drawdown": max_drawdown,
                    "realized_volatility": realized_volatility,
                    "risk_bucket": _risk_bucket(max_drawdown, realized_volatility, self.config),
                }
            )
        return labels


def _group_market_data(market_data: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in market_data:
        ticker = str(row.get("ticker", ""))
        if ticker:
            grouped.setdefault(ticker, []).append(row)
    return {ticker: sorted(rows, key=lambda row: str(row["date"])) for ticker, rows in grouped.items()}


def _return_bucket(forward_return: float, threshold: float) -> str:
    if forward_return >= threshold:
        return "outperform"
    if forward_return <= -threshold:
        return "underperform"
    return "neutral"


def _max_drawdown(values: list[float]) -> float:
    if not values:
        return 0.0
    peak = values[0]
    drawdown = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            drawdown = min(drawdown, (value / peak) - 1.0)
    return drawdown


def _realized_volatility(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    returns = [(values[index] / values[index - 1]) - 1.0 for index in range(1, len(values)) if values[index - 1] > 0]
    if not returns:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / len(returns)
    return math.sqrt(variance)


def _risk_bucket(max_drawdown: float, realized_volatility: float, config: LabelConfig) -> str:
    if max_drawdown <= config.high_risk_drawdown or realized_volatility >= config.high_risk_volatility:
        return "high"
    if max_drawdown <= config.medium_risk_drawdown or realized_volatility >= config.medium_risk_volatility:
        return "medium"
    return "low"
