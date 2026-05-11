"""Decoded examples from the chronological backtest split."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mega_trading.core.store import ArtifactPaths, LocalObjectStore
from mega_trading.dataset import NumpyTickerTimeDataset, prepared_split_counts, split_totals
from mega_trading.events import STREAM_CONTRACT
from mega_trading.tokenizer import MarketEventTokenizer


@dataclass(frozen=True)
class BacktestExamplesResult:
    report_path: str


def run_backtest_examples(
    store: LocalObjectStore,
    run_id: str,
    mixture_name: str = "public",
    max_sequences: int = 3,
    max_tokens: int = 16,
) -> BacktestExamplesResult:
    """Write a small decoded sample of held-out backtest sequences."""
    if max_sequences <= 0:
        raise ValueError("max_sequences must be positive")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    profile = store.read_json(f"datasets/mixture={mixture_name}/tokens-profile.json")
    if profile.get("stream_contract") != STREAM_CONTRACT:
        raise ValueError("profile is not a valid token stream contract")
    numpy_metadata = dict(profile.get("numpy_dataset", {}))
    ticker_counts = {str(key): int(value) for key, value in dict(profile["sequence_counts"]).items()}
    split_counts = prepared_split_counts(numpy_metadata, ticker_counts, validation_fraction=0.0)
    split_count_totals = split_totals(split_counts)
    if split_count_totals["backtest"] <= 0:
        raise ValueError("prepared dataset has no backtest rows")

    tokenizer = MarketEventTokenizer.from_dict(store.read_json(str(profile["tokenizer_path"])))
    dataset = NumpyTickerTimeDataset(store, numpy_metadata, split_counts, split="backtest")
    examples = [
        _example_payload(sequence_index, example["labels"].tolist(), tokenizer, max_tokens)
        for sequence_index, example in enumerate(dataset)
        if sequence_index < max_sequences
    ]
    payload: dict[str, Any] = {
        "stage": "backtest_examples",
        "run_id": run_id,
        "mixture": mixture_name,
        "max_sequences": max_sequences,
        "max_tokens": max_tokens,
        "split_counts": split_count_totals,
        "examples": examples,
    }
    report_path = ArtifactPaths().eval(run_id, "backtest-examples")
    store.write_json(report_path, payload)
    return BacktestExamplesResult(report_path=report_path)


def _example_payload(
    sequence_index: int,
    token_ids: list[int],
    tokenizer: MarketEventTokenizer,
    max_tokens: int,
) -> dict[str, Any]:
    return {
        "sequence_index": sequence_index,
        "tokens": [
            {
                "position": position,
                "token_id": int(token_id),
                "token_name": tokenizer.token_name(int(token_id)),
                "relative_price_bps": tokenizer.relative_price_value(int(token_id)),
                "price_depth_bps": tokenizer.price_depth_value(int(token_id)),
            }
            for position, token_id in enumerate(token_ids[:max_tokens])
        ],
    }
