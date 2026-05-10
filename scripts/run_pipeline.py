"""One-command local and remote training pipelines."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


LOCAL_DATA_DIR = Path(".mega-trading/binance-local")
LOCAL_MIXTURE = "binance_local"
REMOTE_LOCAL_DATA_DIR = Path(".mega-trading/binance-modal")
REMOTE_MIXTURE = "binance_public"
REMOTE_DATA_DIR = "/data/binance-trades"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the simple Mega-Trading local or remote pipeline.")
    parser.add_argument("target", choices=["local", "remote"], help="local trains on this machine; remote uploads data and trains on Modal")
    parser.add_argument("--local-steps", type=int, default=5, help="training steps for local smoke training")
    parser.add_argument("--remote-steps", type=int, default=50_000, help="training steps for Modal training")
    parser.add_argument("--force-data", action="store_true", help="rebuild data even when numpy shards already exist")
    parser.add_argument("--skip-wandb-sync", action="store_true", help="do not sync local W&B login to Modal Secret")
    args = parser.parse_args()

    if args.target == "local":
        _ensure_data(
            data_dir=LOCAL_DATA_DIR,
            mixture=LOCAL_MIXTURE,
            ingest_command=["uv", "run", "mega-trading", "ingest", "--config", "configs/ingest-binance-local.toml"],
            build_command=["uv", "run", "mega-trading", "build", "--config-name", "binance-local"],
            force=args.force_data,
        )
        _run(
            [
                "uv",
                "run",
                "mega-trading",
                "train",
                "--config-name",
                "binance-local",
                f"training.max_steps={args.local_steps}",
                "training.eval_interval=1",
                "training.wandb_enabled=false",
            ]
        )
        return 0

    _ensure_data(
        data_dir=REMOTE_LOCAL_DATA_DIR,
        mixture=REMOTE_MIXTURE,
        ingest_command=["uv", "run", "mega-trading", "ingest", "--config", "configs/ingest-binance-modal-prep.toml"],
        build_command=["uv", "run", "mega-trading", "build", "--config-name", "binance-modal-prep"],
        force=args.force_data,
    )
    _run(["uv", "run", "modal", "volume", "put", "mega-trading-artifacts", str(REMOTE_LOCAL_DATA_DIR), "/binance-trades"])
    if not args.skip_wandb_sync:
        _run(["uv", "run", "python", "scripts/sync_wandb_modal_secret.py"])
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
            "modal-binance",
            "--data-dir",
            REMOTE_DATA_DIR,
            "--strategy",
            "fsdp",
            "--max-steps",
            str(args.remote_steps),
        ]
    )
    return 0


def _ensure_data(
    data_dir: Path,
    mixture: str,
    ingest_command: list[str],
    build_command: list[str],
    force: bool,
) -> None:
    if not force and _shards_exist(data_dir, mixture):
        print(f"data ready at {data_dir} mixture={mixture}; skipping ingest/build")
        return
    _run(ingest_command)
    _run(build_command)


def _shards_exist(data_dir: Path, mixture: str) -> bool:
    shard_root = data_dir / "stage=05_shards" / f"mixture={mixture}"
    required = ("tokens.npy", "ticker_ids.npy", "tokens-numpy.json", "tokens-profile.json", "tokenizer.json")
    return all((shard_root / name).exists() for name in required)


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
