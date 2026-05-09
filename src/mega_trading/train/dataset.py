"""Dataset utilities for stream shards."""

from __future__ import annotations

import torch
from collections.abc import Callable, Iterator

from torch.utils.data import Dataset, IterableDataset

from mega_trading.train.config import RETURN_TO_ID, RISK_TO_ID


class TradingFoundationDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        rows: list[dict[str, object]],
        market_window_size: int,
        news_size: int,
        sec_filing_size: int,
        earnings_size: int,
        macro_size: int,
    ) -> None:
        self.rows = rows
        self.market_window_size = market_window_size
        self.news_size = news_size
        self.sec_filing_size = sec_filing_size
        self.earnings_size = earnings_size
        self.macro_size = macro_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return row_to_item(self.rows[index], self.market_window_size, self.news_size, self.sec_filing_size, self.earnings_size, self.macro_size)


class TradingFoundationIterableDataset(IterableDataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        rows: Callable[[], Iterator[dict[str, object]]],
        market_window_size: int,
        news_size: int,
        sec_filing_size: int,
        earnings_size: int,
        macro_size: int,
    ) -> None:
        self.rows = rows
        self.market_window_size = market_window_size
        self.news_size = news_size
        self.sec_filing_size = sec_filing_size
        self.earnings_size = earnings_size
        self.macro_size = macro_size

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for row in self.rows():
            yield row_to_item(row, self.market_window_size, self.news_size, self.sec_filing_size, self.earnings_size, self.macro_size)


def row_to_item(
    row: dict[str, object],
    market_window_size: int,
    news_size: int,
    sec_filing_size: int,
    earnings_size: int,
    macro_size: int,
) -> dict[str, torch.Tensor]:
    market_data = _market_data_tensor(
        _float_list(row.get("market_returns", [])),
        _float_list(row.get("market_levels", [])),
        market_window_size,
    )
    news = _fixed_vector(_float_list(row.get("news_embeddings", [])), news_size, scale=1.0)
    sec_filings = _fixed_vector(
        _float_list(row.get("sec_filing_features", [])),
        sec_filing_size,
        scale=1_000_000_000.0,
    )
    earnings = _fixed_vector(_float_list(row.get("earnings_features", [])), earnings_size, scale=1.0)
    macro = _fixed_vector(_float_list(row.get("macro_features", [])), macro_size, scale=1.0)
    return {
        "market_data": market_data,
        "news": news,
        "sec_filings": sec_filings,
        "earnings": earnings,
        "macro": macro,
        "return_label": torch.tensor(RETURN_TO_ID[str(row["return_label"])], dtype=torch.long),
        "risk_label": torch.tensor(RISK_TO_ID[str(row["risk_label"])], dtype=torch.long),
    }


def infer_stream_sizes(
    rows: list[dict[str, object]],
    market_window_size: int | None,
    news_size: int | None,
    sec_filing_size: int | None,
    earnings_size: int | None,
    macro_size: int | None,
) -> tuple[int, int, int, int, int]:
    inferred_market_data = max(1, max((len(row.get("market_returns", [])) for row in rows), default=1))
    inferred_news = max(1, max((len(row.get("news_embeddings", [])) for row in rows), default=1))
    inferred_sec_filings = max(1, max((len(row.get("sec_filing_features", [])) for row in rows), default=1))
    inferred_earnings = max(1, max((len(row.get("earnings_features", [])) for row in rows), default=1))
    inferred_macro = max(1, max((len(row.get("macro_features", [])) for row in rows), default=1))
    return (
        market_window_size or inferred_market_data,
        news_size or inferred_news,
        sec_filing_size or inferred_sec_filings,
        earnings_size or inferred_earnings,
        macro_size or inferred_macro,
    )


def _market_data_tensor(returns: list[float], levels: list[float], window_size: int) -> torch.Tensor:
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
