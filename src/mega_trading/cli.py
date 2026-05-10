"""Command line interface for Mega-Trading."""

from __future__ import annotations

import argparse
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from mega_trading.config import BuildConfig, TrainConfig
from mega_trading.core.store import LocalObjectStore
from mega_trading.data.ingest_config import load_ingest_config
from mega_trading.eval import run_eval
from mega_trading.prepare import prepare_numpy_dataset
from mega_trading.trainer import Trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mega-trading", description="Mega-Trading paper-style order-flow CLI.")
    parser.add_argument("--version", action="store_true", help="print the package version and exit")
    subparsers = parser.add_subparsers(dest="command")
    prepare = subparsers.add_parser("prepare", help="prepare partitioned NumPy token shards")
    prepare.add_argument("--ingest-config", required=True, help="path to source TOML config")
    prepare.add_argument("--config-dir", default="configs", help="Hydra config directory")
    prepare.add_argument("--config-name", default="default", help="Hydra config name")
    prepare.add_argument("overrides", nargs="*", help="Hydra overrides for build settings")
    train = subparsers.add_parser("train", help="train decoder-only next-token model")
    train.add_argument("--config-dir", default="configs", help="Hydra config directory")
    train.add_argument("--config-name", default="default", help="Hydra config name")
    train.add_argument("overrides", nargs="*", help="Hydra overrides such as training.max_steps=10 model.hidden_dim=64")
    eval_command = subparsers.add_parser("eval", help="evaluate generated event-feature distributions")
    eval_command.add_argument("--config-dir", default="configs", help="Hydra config directory")
    eval_command.add_argument("--config-name", default="default", help="Hydra config name")
    eval_command.add_argument("--run-id", default=None, help="run id to evaluate")
    eval_command.add_argument("overrides", nargs="*", help="Hydra overrides for data/eval settings")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        from mega_trading import __version__

        print(__version__)
    elif args.command == "prepare":
        config = load_config(Path(args.config_dir), args.config_name, list(args.overrides))
        result = _run_prepare_config(config, Path(args.ingest_config))
        print(f"wrote partition-ready NumPy metadata to {config.data.data_dir}/{result.numpy_metadata_path}")
    elif args.command == "train":
        config = load_config(Path(args.config_dir), args.config_name, list(args.overrides))
        result = _run_train_config(config)
        print(f"wrote training artifacts to {config.data.data_dir}/runs/{config.run.run_id}")
        print(f"manifest: {result.manifest_path}")
    elif args.command == "eval":
        config = load_config(Path(args.config_dir), args.config_name, list(args.overrides))
        result = _run_eval_config(config, args.run_id)
        print(f"wrote eval report to {config.data.data_dir}/{result.report_path}")
    return 0


def load_config(config_dir: Path, config_name: str, overrides: list[str]) -> DictConfig:
    config_root = config_dir if config_dir.is_absolute() else Path.cwd() / config_dir
    with initialize_config_dir(config_dir=str(config_root), version_base=None):
        config = compose(config_name=config_name, overrides=overrides)
    OmegaConf.resolve(config)
    return config


def _run_prepare_config(config: DictConfig, ingest_config_path: Path):
    build_config = BuildConfig(
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
    )
    return prepare_numpy_dataset(load_ingest_config(ingest_config_path), build_config)


def _run_train_config(config: DictConfig):
    store = LocalObjectStore(Path(str(config.data.data_dir)))
    shard_path = f"stage=05_shards/mixture={config.data.mixture}/tokens.npy"
    train_config = TrainConfig(
        run_id=str(config.run.run_id),
        mixture_name=str(config.data.mixture),
        max_steps=int(config.training.max_steps),
        batch_size=int(config.training.batch_size),
        learning_rate=float(config.training.learning_rate),
        validation_fraction=float(config.training.validation_fraction),
        eval_interval=int(config.training.eval_interval),
        max_eval_batches=_optional_int(config.training.max_eval_batches),
        hidden_dim=int(config.model.hidden_dim),
        layers=int(config.model.layers),
        attention_heads=int(config.model.attention_heads),
        kv_heads=_optional_int(config.model.kv_heads),
        intermediate_dim=_optional_int(config.model.intermediate_dim),
        dropout=float(config.model.dropout),
        rope_theta=float(config.model.rope_theta),
        norm_eps=float(config.model.norm_eps),
        seed=int(config.training.seed),
        device=str(config.training.device),
        precision=str(config.training.precision),
        distributed_strategy=str(config.training.distributed_strategy),
        gradient_accumulation_steps=int(config.training.gradient_accumulation_steps),
        compile=bool(config.training.compile),
        compile_mode=str(config.training.compile_mode),
        attention_backend=str(config.training.attention_backend),
        checkpoint_interval=_optional_int(config.training.checkpoint_interval),
        resume_from_checkpoint=_optional_string(config.training.resume_from_checkpoint),
        wandb_enabled=bool(config.training.wandb_enabled),
        wandb_project=str(config.training.wandb_project),
        wandb_entity=_optional_string(config.training.wandb_entity),
        wandb_mode=str(config.training.wandb_mode),
        progress_bar=bool(config.training.progress_bar),
    )
    return Trainer(store, train_config).train(shard_path)


def _run_eval_config(config: DictConfig, run_id: str | None):
    store = LocalObjectStore(Path(str(config.data.data_dir)))
    return run_eval(
        store,
        run_id=run_id or str(config.run.run_id),
        mixture_name=str(config.data.mixture),
        rollouts=int(config.eval.rollouts),
        generated_tokens=int(config.eval.generated_tokens),
        device=str(config.eval.device),
    )


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
