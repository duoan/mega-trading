"""Command line interface for Mega-Trading."""

from __future__ import annotations

import argparse
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from mega_trading.core.store import LocalObjectStore
from mega_trading.config import BuildConfig, TrainConfig
from mega_trading.data.enrich import DataEnricher
from mega_trading.data.ingest import FixtureIngestor, MarketDataIngestRequest, TickerIngestRequest
from mega_trading.data.ingest_config import IngestPipelineConfig, IngestSourceConfig, load_ingest_config
from mega_trading.data.lance_store import LanceTableStore
from mega_trading.data.public.market import StooqClient, StooqMarketDataIngestor, YahooChartClient, YahooMarketDataIngestor
from mega_trading.data.public.sec import SecClient, SecFilingsIngestor
from mega_trading.data.quality import DataQualityChecker
from mega_trading.eval import run_eval
from mega_trading.events import EventBuilder
from mega_trading.trainer import Trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mega-trading",
        description="Mega-Trading data and training CLI.",
    )
    parser.add_argument("--version", action="store_true", help="print the package version and exit")
    subparsers = parser.add_subparsers(dest="command")
    ingest = subparsers.add_parser("ingest", help="run config-driven data ingestion")
    ingest.add_argument("--config", required=True, help="path to ingest TOML config")
    ingest_public = subparsers.add_parser("ingest-public", help="ingest public SEC filings and Yahoo market data")
    ingest_public.add_argument("--tickers", required=True, help="comma-separated ticker symbols")
    ingest_public.add_argument("--start", required=True, help="market data start date YYYY-MM-DD")
    ingest_public.add_argument("--end", required=True, help="market data end date YYYY-MM-DD")
    ingest_public.add_argument("--out", default=".mega-trading/public", help="artifact output directory")
    ingest_public.add_argument("--sec-user-agent", required=True, help="SEC-compliant User-Agent, including contact email")
    build = subparsers.add_parser("build", help="build event/token shards from public data")
    build.add_argument("--config-dir", default="configs", help="Hydra config directory")
    build.add_argument("--config-name", default="default", help="Hydra config name")
    build.add_argument("overrides", nargs="*", help="Hydra overrides such as data.data_dir=.mega-trading/public")
    train = subparsers.add_parser("train", help="train decoder-only next-token model")
    train.add_argument("--config-dir", default="configs", help="Hydra config directory")
    train.add_argument("--config-name", default="default", help="Hydra config name")
    train.add_argument("overrides", nargs="*", help="Hydra overrides such as training.max_steps=10 model.hidden_dim=64")
    eval_command = subparsers.add_parser("eval", help="evaluate stylized facts")
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
    elif args.command == "ingest":
        config = load_ingest_config(Path(args.config))
        _run_ingest_config(config)
    elif args.command == "ingest-public":
        tickers = [ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()]
        config = IngestPipelineConfig(
            output_dir=args.out,
            sec_user_agent=args.sec_user_agent,
            sources=(
                IngestSourceConfig(name="sec_filings", tickers=tuple(tickers)),
                IngestSourceConfig(name="yahoo_market_data", tickers=tuple(tickers), start=args.start, end=args.end),
            ),
        )
        _run_ingest_config(config)
    elif args.command == "build":
        config = load_config(Path(args.config_dir), args.config_name, list(args.overrides))
        result = _run_build_config(config)
        print(f"wrote token shard to {config.data.data_dir}/{result.shard_path}")
        print(f"profile: {result.profile_path}")
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
    return _load_hydra_config(config_dir, config_name, overrides)


def _load_hydra_config(config_dir: Path, config_name: str, overrides: list[str]) -> DictConfig:
    config_root = config_dir if config_dir.is_absolute() else Path.cwd() / config_dir
    with initialize_config_dir(config_dir=str(config_root), version_base=None):
        config = compose(config_name=config_name, overrides=overrides)
    OmegaConf.resolve(config)
    return config


def _run_build_config(config: DictConfig):
    store = LocalObjectStore(Path(str(config.data.data_dir)))
    build_config = BuildConfig(
        mixture_name=str(config.data.mixture),
        source=str(config.data.source),
        block_size=int(config.build.block_size),
        stride=int(config.build.stride),
        min_events_per_ticker=int(config.build.min_events_per_ticker),
        max_tickers=_optional_int(config.build.max_tickers),
    )
    return EventBuilder(store, build_config).build()


def _run_train_config(config: DictConfig):
    store = LocalObjectStore(Path(str(config.data.data_dir)))
    shard_path = f"stage=05_shards/mixture={config.data.mixture}/tokens.jsonl"
    train_config = TrainConfig(
        run_id=str(config.run.run_id),
        mixture_name=str(config.data.mixture),
        max_steps=int(config.training.max_steps),
        batch_size=int(config.training.batch_size),
        learning_rate=float(config.training.learning_rate),
        validation_fraction=float(config.training.validation_fraction),
        eval_interval=int(config.training.eval_interval),
        hidden_dim=int(config.model.hidden_dim),
        layers=int(config.model.layers),
        attention_heads=int(config.model.attention_heads),
        dropout=float(config.model.dropout),
        seed=int(config.training.seed),
        device=str(config.training.device),
        precision=str(config.training.precision),
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


def _run_ingest_config(config: IngestPipelineConfig) -> None:
    store = LocalObjectStore(Path(config.output_dir))
    table_store = LanceTableStore(Path(config.output_dir) / "lancedb")
    normalization_manifests: list[str] = []
    for source in config.enabled_sources():
        if source.name == "fixture":
            result = FixtureIngestor(store, table_store=table_store).ingest()
        elif source.name == "sec_filings":
            if not config.sec_user_agent:
                raise ValueError("sec_filings requires ingest.sec_user_agent")
            result = SecFilingsIngestor(store, SecClient(user_agent=config.sec_user_agent), table_store=table_store).ingest(
                TickerIngestRequest(tickers=source.tickers)
            )
        elif source.name == "yahoo_market_data":
            result = YahooMarketDataIngestor(store, YahooChartClient(), table_store=table_store).ingest(
                MarketDataIngestRequest(tickers=source.tickers, start=str(source.start), end=str(source.end))
            )
        elif source.name == "stooq_market_data":
            result = StooqMarketDataIngestor(store, StooqClient(), table_store=table_store).ingest(
                MarketDataIngestRequest(tickers=source.tickers, start=str(source.start), end=str(source.end))
            )
        else:
            raise ValueError(f"unsupported ingest source: {source.name}")
        normalization_manifests.append(result.normalization_manifest_path)
    quality_passed = True
    if config.quality_enabled:
        quality_result = DataQualityChecker(store).run(
            normalization_manifests,
            run_id="configured-ingest",
            fail_on_error=config.quality_fail_on_error,
        )
        quality_passed = quality_result.passed
    if config.enrichment_enabled and quality_passed:
        DataEnricher(store).run(run_id="configured-ingest")
    print(f"wrote configured ingest artifacts to {config.output_dir}")
