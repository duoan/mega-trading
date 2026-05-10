"""One-command local and remote training pipelines."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


LOCAL_DATA_DIR = Path(".mega-trading/binance-local")
LOCAL_MIXTURE = "binance_local"
LOCAL_MIN_SEQUENCES = 50_000
REMOTE_LOCAL_DATA_DIR = Path(".mega-trading/binance-modal")
REMOTE_MIXTURE = "binance_public"
REMOTE_DATA_DIR = "/data/binance-trades"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the simple Mega-Trading local or remote pipeline.")
    parser.add_argument("target", choices=["local", "remote"], help="local trains on this machine; remote uploads data and trains on Modal")
    parser.add_argument("--local-steps", type=int, default=300, help="training steps for local training")
    parser.add_argument("--remote-steps", type=int, default=50_000, help="training steps for Modal training")
    parser.add_argument("--force-data", action="store_true", help="rebuild data even when numpy shards already exist")
    parser.add_argument("--skip-wandb-sync", action="store_true", help="do not sync local W&B login to Modal Secret")
    args = parser.parse_args()

    if args.target == "local":
        _ensure_data(
            data_dir=LOCAL_DATA_DIR,
            mixture=LOCAL_MIXTURE,
            prepare_command=[
                "uv",
                "run",
                "python",
                "scripts/prepare_numpy_dataset.py",
                "--ingest-config",
                "configs/ingest-binance-local.toml",
                "--config-name",
                "binance-local",
            ],
            min_sequences=LOCAL_MIN_SEQUENCES,
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
                "training.eval_interval=50",
                "training.wandb_enabled=false",
            ]
        )
        return 0

    _ensure_data(
        data_dir=REMOTE_LOCAL_DATA_DIR,
        mixture=REMOTE_MIXTURE,
        prepare_command=[
            "uv",
            "run",
            "python",
            "scripts/prepare_numpy_dataset.py",
            "--ingest-config",
            "configs/ingest-binance-modal-prep.toml",
            "--config-name",
            "binance-modal-prep",
        ],
        min_sequences=0,
        force=args.force_data,
    )
    _run(
        [
            "uv",
            "run",
            "modal",
            "volume",
            "put",
            "mega-trading-artifacts",
            str(REMOTE_LOCAL_DATA_DIR / "datasets"),
            "/binance-trades/datasets",
        ]
    )
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


if __name__ == "__main__":
    raise SystemExit(main())
