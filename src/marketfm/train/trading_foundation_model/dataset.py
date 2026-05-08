"""Dataset utilities for TradingFoundationModel stream shards."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from marketfm.train.trading_foundation_model.config import RETURN_TO_ID, RISK_TO_ID


class TradingFoundationDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        rows: list[dict[str, object]],
        price_window_size: int,
        fundamental_size: int,
        evidence_size: int,
    ) -> None:
        self.rows = rows
        self.price_window_size = price_window_size
        self.fundamental_size = fundamental_size
        self.evidence_size = evidence_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        price = _price_tensor(
            _float_list(row.get("price_returns", [])),
            _float_list(row.get("price_levels", [])),
            self.price_window_size,
        )
        fundamentals = _fixed_vector(
            _float_list(row.get("fundamental_values", [])),
            self.fundamental_size,
            scale=1_000_000_000.0,
        )
        evidence = _fixed_vector(_float_list(row.get("evidence_token_ids", [])), self.evidence_size, scale=10_000.0)
        return {
            "price": price,
            "fundamentals": fundamentals,
            "evidence": evidence,
            "return_label": torch.tensor(RETURN_TO_ID[str(row["return_label"])], dtype=torch.long),
            "risk_label": torch.tensor(RISK_TO_ID[str(row["risk_label"])], dtype=torch.long),
        }


def infer_stream_sizes(
    rows: list[dict[str, object]],
    price_window_size: int | None,
    fundamental_size: int | None,
    evidence_size: int | None,
) -> tuple[int, int, int]:
    inferred_price = max(1, max((len(row.get("price_returns", [])) for row in rows), default=1))
    inferred_fundamentals = max(1, max((len(row.get("fundamental_values", [])) for row in rows), default=1))
    inferred_evidence = max(1, max((len(row.get("evidence_token_ids", [])) for row in rows), default=1))
    return price_window_size or inferred_price, fundamental_size or inferred_fundamentals, evidence_size or inferred_evidence


def _price_tensor(returns: list[float], levels: list[float], window_size: int) -> torch.Tensor:
    fixed_returns = _pad_or_truncate(returns, window_size)
    fixed_levels = _pad_or_truncate(levels, window_size)
    return torch.tensor(list(zip(fixed_returns, fixed_levels)), dtype=torch.float32)


def _fixed_vector(values: list[float], size: int, scale: float) -> torch.Tensor:
    return torch.tensor([value / scale for value in _pad_or_truncate(values, size)], dtype=torch.float32)


def _pad_or_truncate(values: list[float], size: int) -> list[float]:
    values = values[-size:]
    return [0.0] * (size - len(values)) + values


def _float_list(value: object) -> list[float]:
    if not isinstance(value, list):
        return []
    return [float(item) for item in value]
