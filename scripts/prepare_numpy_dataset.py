"""Prepare training-ready NumPy shards without durable intermediate datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from mega_trading.cli import load_config
from mega_trading.config import BuildConfig
from mega_trading.data.ingest_config import load_ingest_config
from mega_trading.prepare import prepare_numpy_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare partitioned NumPy token shards directly from source data.")
    parser.add_argument("--ingest-config", required=True, help="ingest TOML config")
    parser.add_argument("--config-dir", default="configs", help="Hydra config directory")
    parser.add_argument("--config-name", required=True, help="Hydra config name")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides")
    args = parser.parse_args()

    config = load_config(Path(args.config_dir), args.config_name, list(args.overrides))
    result = prepare_numpy_dataset(
        load_ingest_config(Path(args.ingest_config)),
        BuildConfig(
            mixture_name=str(config.data.mixture),
            source=str(config.data.source),
            block_size=int(config.build.block_size),
            stride=int(config.build.stride),
            min_events_per_ticker=int(config.build.min_events_per_ticker),
            max_tickers=_optional_int(config.build.max_tickers),
            tokenizer_method=str(config.build.tokenizer_method),
            tokenizer_clip_quantile=float(config.build.tokenizer_clip_quantile),
            tokenizer_relative_price_bins=int(config.build.tokenizer_relative_price_bins),
            tokenizer_price_bins=int(config.build.tokenizer_price_bins),
            tokenizer_size_bins=int(config.build.tokenizer_size_bins),
            tokenizer_time_bins=int(config.build.tokenizer_time_bins),
            numpy_partition_rows=_optional_int(config.build.numpy_partition_rows),
        ),
    )
    print(f"wrote partition-ready NumPy metadata to {config.data.data_dir}/{result.numpy_metadata_path}")
    return 0


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


if __name__ == "__main__":
    raise SystemExit(main())
