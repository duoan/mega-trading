"""Autoregressive token datasets for training."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import IterableDataset

from mega_trading.core.store import LocalObjectStore


class TokenDataset(IterableDataset[dict[str, torch.Tensor]]):
    """Stream token rows as next-token prediction examples."""

    def __init__(self, store: LocalObjectStore, shard_path: str, start: int = 0, stop: int | None = None) -> None:
        super().__init__()
        self.store = store
        self.shard_path = shard_path
        self.start = start
        self.stop = stop

    def __iter__(self):
        for index, row in enumerate(self.store.iter_jsonl(self.shard_path)):
            if index < self.start:
                continue
            if self.stop is not None and index >= self.stop:
                break
            yield row_to_example(row)


class TickerTimeDataset(IterableDataset[dict[str, torch.Tensor]]):
    """Stream each ticker's early rows for train and late rows for validation."""

    def __init__(
        self,
        store: LocalObjectStore,
        shard_path: str,
        train_counts: dict[str, int],
        split: str,
    ) -> None:
        super().__init__()
        if split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        self.store = store
        self.shard_path = shard_path
        self.train_counts = train_counts
        self.split = split

    def __iter__(self):
        seen: Counter[str] = Counter()
        for row in self.store.iter_jsonl(self.shard_path):
            ticker = str(row["ticker"])
            index = seen[ticker]
            seen[ticker] += 1
            is_train = index < self.train_counts.get(ticker, 0)
            if (self.split == "train" and is_train) or (self.split == "validation" and not is_train):
                yield row_to_example(row)


class NumpyTickerTimeDataset(IterableDataset[dict[str, torch.Tensor]]):
    """Memory-map prebuilt token arrays and split rows by ticker time order."""

    def __init__(
        self,
        store: LocalObjectStore,
        numpy_metadata: dict[str, Any],
        train_counts: dict[str, int],
        split: str,
    ) -> None:
        super().__init__()
        if split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        if numpy_metadata.get("format") != "mega-trading-numpy-token-v1":
            raise ValueError("unsupported numpy token dataset format")
        self.store = store
        self.numpy_metadata = numpy_metadata
        self.train_counts = train_counts
        self.split = split
        ticker_to_id = {str(ticker): int(ticker_id) for ticker, ticker_id in dict(numpy_metadata["ticker_to_id"]).items()}
        self.train_counts_by_id = {ticker_to_id[ticker]: count for ticker, count in train_counts.items() if ticker in ticker_to_id}

    def __iter__(self):
        tokens = np.load(_artifact_target(self.store, str(self.numpy_metadata["tokens_path"])), mmap_mode="r")
        ticker_ids = np.load(_artifact_target(self.store, str(self.numpy_metadata["ticker_ids_path"])), mmap_mode="r")
        seen: Counter[int] = Counter()
        for row_index in range(int(tokens.shape[0])):
            ticker_id = int(ticker_ids[row_index])
            index = seen[ticker_id]
            seen[ticker_id] += 1
            is_train = index < self.train_counts_by_id.get(ticker_id, 0)
            if (self.split == "train" and is_train) or (self.split == "validation" and not is_train):
                yield tokens_to_example(tokens[row_index])


def row_to_example(row: dict[str, Any]) -> dict[str, torch.Tensor]:
    tokens = [int(token) for token in row["tokens"]]
    return tokens_to_example(tokens)


def tokens_to_example(tokens: Iterable[int]) -> dict[str, torch.Tensor]:
    tokens = [int(token) for token in tokens]
    if len(tokens) < 2:
        raise ValueError("token row must contain at least two tokens")
    input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
    labels = torch.tensor(tokens[1:], dtype=torch.long)
    return {"input_ids": input_ids, "labels": labels}


def split_counts(total_rows: int, validation_fraction: float) -> tuple[int, int]:
    if total_rows <= 0:
        raise ValueError("total_rows must be positive")
    if validation_fraction <= 0.0:
        return total_rows, 0
    validation_rows = max(1, int(total_rows * validation_fraction))
    validation_rows = min(validation_rows, total_rows - 1)
    return total_rows - validation_rows, validation_rows


def per_ticker_train_counts(ticker_counts: dict[str, int], validation_fraction: float) -> dict[str, int]:
    train_counts: dict[str, int] = {}
    for ticker, count in ticker_counts.items():
        if count <= 0:
            train_counts[ticker] = 0
        elif validation_fraction <= 0.0 or count == 1:
            train_counts[ticker] = count
        else:
            validation_rows = max(1, int(count * validation_fraction))
            validation_rows = min(validation_rows, count - 1)
            train_counts[ticker] = count - validation_rows
    return train_counts


def cycle_batches(loader: Iterable[dict[str, torch.Tensor]]):
    while True:
        yielded = False
        for batch in loader:
            yielded = True
            yield batch
        if not yielded:
            raise ValueError("training dataset is empty")


def _artifact_target(store: LocalObjectStore, path: str) -> Path:
    target = Path(path)
    if target.is_absolute() or ".." in target.parts:
        raise ValueError(f"artifact path must be relative and safe: {path}")
    return store.root / target
