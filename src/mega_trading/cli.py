"""Command line interface for Mega-Trading."""

from __future__ import annotations

import argparse
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from mega_trading.core.store import LocalObjectStore
from mega_trading.data.enrich import DataEnricher
from mega_trading.data.ingest import PriceIngestRequest, TickerIngestRequest
from mega_trading.data.ingest_config import IngestPipelineConfig, IngestSourceConfig, load_ingest_config
from mega_trading.data.labels import LabelConfig
from mega_trading.data.lance_store import LanceTableStore
from mega_trading.data.public.prices import StooqClient, StooqPriceIngestor, YahooChartClient, YahooPriceIngestor
from mega_trading.data.public.sec import SecClient, SecCompanyFactsIngestor
from mega_trading.data.quality import DataQualityChecker
from mega_trading.data.samples import MultiStreamSampleBuilder
from mega_trading.data.tokenize import StreamShardBuilder
from mega_trading.train.config import TradingFoundationTrainConfig
from mega_trading.train.trainer import TradingFoundationTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mega-trading",
        description="Mega-Trading data, model, and training CLI.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="print the package version and exit",
    )
    subparsers = parser.add_subparsers(dest="command")
    ingest = subparsers.add_parser("ingest", help="run config-driven data ingestion")
    ingest.add_argument("--config", required=True, help="path to ingest TOML config")
    ingest_public = subparsers.add_parser("ingest-public", help="ingest public SEC fundamentals and Yahoo prices")
    ingest_public.add_argument("--tickers", required=True, help="comma-separated ticker symbols")
    ingest_public.add_argument("--start", required=True, help="price start date YYYY-MM-DD")
    ingest_public.add_argument("--end", required=True, help="price end date YYYY-MM-DD")
    ingest_public.add_argument("--out", default=".mega-trading/public", help="artifact output directory")
    ingest_public.add_argument("--sec-user-agent", required=True, help="SEC-compliant User-Agent, including contact email")
    train = subparsers.add_parser("train", help="run TradingFoundationModel training from Hydra config")
    train.add_argument(
        "--config-dir",
        default="configs/train",
        help="Hydra config directory for training ablations",
    )
    train.add_argument("--config-name", default="default", help="Hydra config name")
    train.add_argument("overrides", nargs="*", help="Hydra overrides such as training.max_steps=100 model.hidden_dim=64")
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
                IngestSourceConfig(name="sec_companyfacts", tickers=tuple(tickers)),
                IngestSourceConfig(name="yahoo_prices", tickers=tuple(tickers), start=args.start, end=args.end),
            ),
        )
        _run_ingest_config(config)
    elif args.command == "train":
        config = load_train_config(Path(args.config_dir), args.config_name, list(args.overrides))
        result = _run_train_config(config)
        print(f"wrote TradingFoundationModel training artifacts to {config.data.data_dir}/runs/{config.run.run_id}")
        print(f"manifest: {result.manifest_path}")
    return 0


def load_train_config(config_dir: Path, config_name: str, overrides: list[str]) -> DictConfig:
    config_root = config_dir if config_dir.is_absolute() else Path.cwd() / config_dir
    with initialize_config_dir(config_dir=str(config_root), version_base=None):
        config = compose(config_name=config_name, overrides=overrides)
    OmegaConf.resolve(config)
    return config


def _run_train_config(config: DictConfig):
    store = LocalObjectStore(Path(str(config.data.data_dir)))
    shard_path = f"stage=05_shards/mixture={config.data.mixture}/samples.jsonl"
    train_config = TradingFoundationTrainConfig(
        run_id=str(config.run.run_id),
        max_steps=int(config.training.max_steps),
        hidden_dim=int(config.model.hidden_dim),
        batch_size=int(config.training.batch_size),
        learning_rate=float(config.training.learning_rate),
        price_window_size=_optional_int(config.model.price_window_size),
        fundamental_size=_optional_int(config.model.fundamental_size),
        evidence_size=_optional_int(config.model.evidence_size),
        seed=int(config.training.seed),
        device=str(config.training.device),
    )
    return TradingFoundationTrainer(store, train_config).train(shard_path)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _run_ingest_config(config: IngestPipelineConfig) -> None:
    store = LocalObjectStore(Path(config.output_dir))
    table_store = LanceTableStore(Path(config.output_dir) / "lancedb")
    normalization_manifests: list[str] = []
    for source in config.enabled_sources():
        if source.name == "sec_companyfacts":
            if not config.sec_user_agent:
                raise ValueError("sec_companyfacts requires ingest.sec_user_agent")
            result = SecCompanyFactsIngestor(store, SecClient(user_agent=config.sec_user_agent), table_store=table_store).ingest(
                TickerIngestRequest(tickers=source.tickers)
            )
        elif source.name == "yahoo_prices":
            result = YahooPriceIngestor(store, YahooChartClient(), table_store=table_store).ingest(
                PriceIngestRequest(tickers=source.tickers, start=str(source.start), end=str(source.end))
            )
        elif source.name == "stooq_prices":
            result = StooqPriceIngestor(store, StooqClient(), table_store=table_store).ingest(
                PriceIngestRequest(tickers=source.tickers, start=str(source.start), end=str(source.end))
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
    if config.training_data_enabled and quality_passed:
        MultiStreamSampleBuilder(
            store,
            LabelConfig(
                input_window_observations=config.training_input_window_observations,
                horizon_observations=config.training_horizon_observations,
                return_threshold=config.training_return_threshold,
            ),
        ).build(mixture_name=config.training_mixture_name, run_id="configured-ingest")
        StreamShardBuilder(store).build(config.training_mixture_name)
    print(f"wrote configured ingest artifacts to {config.output_dir}")
