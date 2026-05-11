"""One-command Mac, RTX, and Modal training pipelines."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


MODAL_VOLUME_NAME = "mega-trading-artifacts"
MODAL_VOLUME_DATA_PREFIX = "/shared"
MODAL_RUNTIME_DATA_DIR = "/data/shared"


@dataclass(frozen=True)
class Environment:
    name: str
    data_dir: Path
    ingest_config: str
    min_sequences: int = 0

    @property
    def config_name(self) -> str:
        return self.name

    @property
    def mixture(self) -> str:
        return self.name


SHARED_DATA_DIR = Path(".mega-trading/data")
MAC = Environment("mac", SHARED_DATA_DIR, "configs/ingest-mac.toml", min_sequences=50_000)
RTX = Environment("rtx", SHARED_DATA_DIR, "configs/ingest-rtx.toml")
MODAL = Environment("modal", SHARED_DATA_DIR, "configs/ingest-modal.toml")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Mega-Trading environment pipeline.")
    parser.add_argument(
        "target",
        choices=["mac", "rtx", "modal"],
        help="environment to run end to end",
    )
    parser.add_argument("--mac-steps", type=int, default=300, help="training steps for the mac environment")
    parser.add_argument("--mac-backtest-batches", type=int, default=32, help="maximum mac backtest batches to score")
    parser.add_argument("--rtx-backtest-batches", type=int, default=128, help="maximum rtx backtest batches to score")
    parser.add_argument("--modal-steps", type=int, default=50_000, help="training steps for the Modal environment")
    parser.add_argument("--modal-backtest-batches", type=int, default=128, help="maximum modal backtest batches to score")
    parser.add_argument("--force-data", action="store_true", help="rebuild data even when numpy shards already exist")
    parser.add_argument("--skip-artifact-sync", action="store_true", help="do not download Modal training outputs")
    parser.add_argument("--mlflow-tracking-uri", help="MLflow tracking server URI for local training runs")
    args = parser.parse_args()
    mlflow_overrides = _mlflow_overrides(args.mlflow_tracking_uri)

    if args.target == "mac":
        _run_environment(
            MAC,
            train_overrides=[
                f"training.max_steps={args.mac_steps}",
                "training.eval_interval=50",
                *mlflow_overrides,
            ],
            backtest_batches=args.mac_backtest_batches,
            force_data=args.force_data,
        )
        return 0

    if args.target == "rtx":
        _run_environment(
            RTX,
            train_overrides=mlflow_overrides,
            backtest_batches=args.rtx_backtest_batches,
            force_data=args.force_data,
        )
        return 0

    _run_modal(args)
    return 0


def _run_environment(
    environment: Environment,
    train_overrides: list[str],
    backtest_batches: int,
    force_data: bool,
    command_overrides: list[str] | None = None,
) -> None:
    command_overrides = command_overrides or []
    _ensure_data(
        data_dir=environment.data_dir,
        mixture=environment.mixture,
        prepare_command=_prepare_command(environment, command_overrides),
        min_sequences=environment.min_sequences,
        force=force_data,
    )
    _run(["uv", "run", "mega-trading", "train", "--config-name", environment.config_name, *command_overrides, *train_overrides])
    _run(
        [
            "uv",
            "run",
            "mega-trading",
            "backtest",
            "--config-name",
            environment.config_name,
            "--max-batches",
            str(backtest_batches),
            *command_overrides,
        ]
    )
    _run(["uv", "run", "mega-trading", "report", "--config-name", environment.config_name, *command_overrides])


def _run_modal(args: argparse.Namespace) -> None:
    staging_overrides = [f"data.data_dir={MODAL.data_dir}", "eval.device=auto"]
    _ensure_data(
        data_dir=MODAL.data_dir,
        mixture=MODAL.mixture,
        prepare_command=_prepare_command(MODAL, staging_overrides),
        min_sequences=MODAL.min_sequences,
        force=args.force_data,
    )
    _run(
        [
            "uv",
            "run",
            "modal",
            "volume",
            "put",
            MODAL_VOLUME_NAME,
            str(MODAL.data_dir / "datasets"),
            f"{MODAL_VOLUME_DATA_PREFIX}/datasets",
        ]
    )
    _run(
        [
            "uv",
            "run",
            "modal",
            "run",
            "modal_train.py",
            "--mode",
            "cluster",
            "--run-id",
            MODAL.name,
            "--config-name",
            MODAL.config_name,
            "--data-dir",
            MODAL_RUNTIME_DATA_DIR,
            "--strategy",
            "fsdp",
            "--max-steps",
            str(args.modal_steps),
        ]
    )
    if not args.skip_artifact_sync:
        _sync_modal_artifacts(MODAL.data_dir)
        _run(
            [
                "uv",
                "run",
                "mega-trading",
                "backtest",
                "--config-name",
                MODAL.config_name,
                "--max-batches",
                str(args.modal_backtest_batches),
                *staging_overrides,
            ]
        )
        _run(["uv", "run", "mega-trading", "report", "--config-name", MODAL.config_name, *staging_overrides])


def _prepare_command(environment: Environment, overrides: list[str] | None = None) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        "scripts/prepare_numpy_dataset.py",
        "--ingest-config",
        environment.ingest_config,
        "--config-name",
        environment.config_name,
        *(overrides or []),
    ]


def _mlflow_overrides(tracking_uri: str | None) -> list[str]:
    if not tracking_uri:
        return []
    return [f"training.mlflow_tracking_uri={tracking_uri}"]


def _ensure_data(
    data_dir: Path,
    mixture: str,
    prepare_command: list[str],
    min_sequences: int,
    force: bool,
) -> None:
    if not force and _shards_exist(data_dir, mixture, min_sequences=min_sequences):
        print(f"data ready at {data_dir} mixture={mixture}; skipping prepare")
        return
    _run(prepare_command)


def _shards_exist(data_dir: Path, mixture: str, min_sequences: int = 0) -> bool:
    dataset_root = data_dir / "datasets" / f"mixture={mixture}"
    metadata_path = dataset_root / "tokens-numpy.json"
    required = (metadata_path, dataset_root / "tokens-profile.json", dataset_root / "tokenizer.json")
    if not all(path.exists() for path in required):
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("sequence_count", 0)) < min_sequences:
        return False
    splits = metadata.get("splits", {})
    totals = splits.get("totals", {}) if isinstance(splits, dict) else {}
    if int(totals.get("train", 0)) <= 0 or int(totals.get("backtest", 0)) <= 0:
        return False
    if metadata.get("storage") == "token_stream":
        if not metadata.get("partitions"):
            return False
        return all((data_dir / str(partition["tokens_path"])).exists() for partition in metadata.get("partitions", []))
    if metadata.get("partitioned"):
        if not metadata.get("partitions"):
            return False
        return all(
            (data_dir / str(partition["tokens_path"])).exists() and (data_dir / str(partition["ticker_ids_path"])).exists()
            for partition in metadata.get("partitions", [])
        )
    return (dataset_root / "tokens.npy").exists() and (dataset_root / "ticker_ids.npy").exists()


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, check=True)


def _sync_modal_artifacts(data_dir: Path) -> None:
    """Download Modal training outputs that are produced inside the shared data volume."""
    for name in ("runs", "manifests"):
        destination = data_dir / name
        destination.mkdir(parents=True, exist_ok=True)
        _run(
            [
                "uv",
                "run",
                "modal",
                "volume",
                "get",
                "--force",
                MODAL_VOLUME_NAME,
                f"{MODAL_VOLUME_DATA_PREFIX}/{name}",
                str(destination),
            ]
        )


if __name__ == "__main__":
    raise SystemExit(main())
