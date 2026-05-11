"""Autoregressive token datasets for training."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, MutableMapping

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from mega_trading.core.store import LocalObjectStore


class NumpyTickerTimeDataset(IterableDataset[dict[str, torch.Tensor]]):
    """Memory-map prebuilt token arrays and split rows by prepared ticker-time splits."""

    def __init__(
        self,
        store: LocalObjectStore,
        numpy_metadata: dict[str, Any],
        split_counts: dict[str, Any],
        split: str,
        preload_numpy_arrays: bool = False,
        array_cache: MutableMapping[str, np.ndarray] | None = None,
    ) -> None:
        super().__init__()
        if split not in {"train", "validation", "backtest"}:
            raise ValueError("split must be train, validation, or backtest")
        if numpy_metadata.get("format") != "mega-trading-numpy-token-v1":
            raise ValueError("unsupported numpy token dataset format")
        self.store = store
        self.numpy_metadata = numpy_metadata
        self.split_counts = split_counts
        self.split = split
        self.preload_numpy_arrays = preload_numpy_arrays
        self.array_cache = array_cache if array_cache is not None else {}
        ticker_to_id = {str(ticker): int(ticker_id) for ticker, ticker_id in dict(numpy_metadata["ticker_to_id"]).items()}
        self.split_counts_by_id = {
            ticker_to_id[ticker]: {
                "train": int(counts.get("train", 0)),
                "validation": int(counts.get("validation", 0)),
                "backtest": int(counts.get("backtest", 0)),
            }
            for ticker, counts in split_counts.items()
            if ticker in ticker_to_id
        }
        if self.preload_numpy_arrays:
            self._preload_arrays()

    def __iter__(self):
        if self.numpy_metadata.get("storage") == "token_stream":
            yield from self._iter_token_streams()
            return
        if bool(self.numpy_metadata.get("partitioned")):
            yield from self._iter_partitions()
            return
        tokens = self._load_array(str(self.numpy_metadata["tokens_path"]))
        ticker_ids = self._load_array(str(self.numpy_metadata["ticker_ids_path"]))
        seen: Counter[int] = Counter()
        global_row_index = 0
        for row_index in range(int(tokens.shape[0])):
            ticker_id = int(ticker_ids[row_index])
            index = seen[ticker_id]
            seen[ticker_id] += 1
            if _row_in_split(index, self.split_counts_by_id.get(ticker_id, {}), self.split) and _worker_accepts(
                global_row_index
            ):
                yield tokens_to_example(tokens[row_index])
            global_row_index += 1

    def _iter_partitions(self):
        seen: Counter[int] = Counter()
        global_row_index = 0
        for partition in self.numpy_metadata.get("partitions", []):
            tokens = self._load_array(str(partition["tokens_path"]))
            ticker_ids = self._load_array(str(partition["ticker_ids_path"]))
            for row_index in range(int(tokens.shape[0])):
                ticker_id = int(ticker_ids[row_index])
                index = seen[ticker_id]
                seen[ticker_id] += 1
                if _row_in_split(index, self.split_counts_by_id.get(ticker_id, {}), self.split) and _worker_accepts(
                    global_row_index
                ):
                    yield tokens_to_example(tokens[row_index])
                global_row_index += 1

    def _iter_token_streams(self):
        sequence_length = int(self.numpy_metadata["sequence_length"])
        stride = int(self.numpy_metadata["stride"])
        seen: Counter[int] = Counter()
        global_row_index = 0
        for partition in self.numpy_metadata.get("partitions", []):
            ticker_id = int(partition["ticker_id"])
            counts = self.split_counts_by_id.get(ticker_id, {})
            tokens = self._load_array(str(partition["tokens_path"]))
            for _partition_index in range(int(partition["sequence_count"])):
                index = seen[ticker_id]
                seen[ticker_id] += 1
                if not _row_in_split(index, counts, self.split):
                    global_row_index += 1
                    continue
                if not _worker_accepts(global_row_index):
                    global_row_index += 1
                    continue
                offset = _partition_index * stride
                yield tokens_to_example(tokens[offset : offset + sequence_length])
                global_row_index += 1

    def _preload_arrays(self) -> None:
        if self.numpy_metadata.get("storage") == "token_stream":
            for partition in self.numpy_metadata.get("partitions", []):
                self._load_array(str(partition["tokens_path"]))
            return
        if bool(self.numpy_metadata.get("partitioned")):
            for partition in self.numpy_metadata.get("partitions", []):
                self._load_array(str(partition["tokens_path"]))
                self._load_array(str(partition["ticker_ids_path"]))
            return
        self._load_array(str(self.numpy_metadata["tokens_path"]))
        self._load_array(str(self.numpy_metadata["ticker_ids_path"]))

    def _load_array(self, path: str) -> np.ndarray:
        target = _artifact_target(self.store, path)
        if not self.preload_numpy_arrays:
            return np.load(target, mmap_mode="r")
        cache_key = str(target)
        if cache_key not in self.array_cache:
            self.array_cache[cache_key] = np.load(target)
        return self.array_cache[cache_key]


def tokens_to_example(tokens: Iterable[int]) -> dict[str, torch.Tensor]:
    tokens = [int(token) for token in tokens]
    if len(tokens) < 2:
        raise ValueError("token row must contain at least two tokens")
    input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
    labels = torch.tensor(tokens[1:], dtype=torch.long)
    return {"input_ids": input_ids, "labels": labels}


def _worker_accepts(row_index: int) -> bool:
    worker = get_worker_info()
    if worker is None:
        return True
    return row_index % int(worker.num_workers) == int(worker.id)


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


def prepared_split_counts(
    numpy_metadata: dict[str, Any],
    ticker_counts: dict[str, int],
    validation_fraction: float,
) -> dict[str, dict[str, int]]:
    splits = numpy_metadata.get("splits")
    if isinstance(splits, dict) and isinstance(splits.get("counts"), dict):
        return {
            str(ticker): {
                "train": int(counts.get("train", 0)),
                "validation": int(counts.get("validation", 0)),
                "backtest": int(counts.get("backtest", 0)),
            }
            for ticker, counts in dict(splits["counts"]).items()
            if isinstance(counts, dict)
        }
    train_counts = per_ticker_train_counts(ticker_counts, validation_fraction)
    return {
        ticker: {
            "train": train_count,
            "validation": max(int(ticker_counts[ticker]) - train_count, 0),
            "backtest": 0,
        }
        for ticker, train_count in train_counts.items()
    }


def split_totals(split_counts: dict[str, dict[str, int]]) -> dict[str, int]:
    return {
        split: sum(int(counts.get(split, 0)) for counts in split_counts.values())
        for split in ("train", "validation", "backtest")
    }


def _row_in_split(index: int, counts: dict[str, int], split: str) -> bool:
    train = int(counts.get("train", 0))
    validation = int(counts.get("validation", 0))
    backtest = int(counts.get("backtest", 0))
    if split == "train":
        return index < train
    if split == "validation":
        return train <= index < train + validation
    return train + validation <= index < train + validation + backtest


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
