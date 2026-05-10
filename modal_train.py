"""Modal launchers for Mega-Trading GPU and multi-node training."""

from __future__ import annotations

import subprocess
from typing import Literal

import modal
import modal.experimental

APP_NAME = "mega-trading-training"
REPO_DIR = "/workspace/mega-finance"
DATA_VOLUME_PATH = "/data"
DEFAULT_CONFIG_NAME = "modal-binance"
DEFAULT_DATA_DIR = f"{DATA_VOLUME_PATH}/binance-trades"
PAPER_CONFIG_NAME = "modal-paper"
PAPER_DATA_DIR = f"{DATA_VOLUME_PATH}/hf-1m-paper"
DEFAULT_GPU = "H100:8"
WANDB_SECRET_NAME = "wandb-secret"

volume = modal.Volume.from_name("mega-trading-artifacts", create_if_missing=True)
wandb_secret = modal.Secret.from_name(WANDB_SECRET_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("uv")
    .add_local_dir(".", remote_path=REPO_DIR)
    .run_commands(f"cd {REPO_DIR} && uv sync --frozen")
)

app = modal.App(APP_NAME, image=image)


def _training_overrides(
    config_name: str,
    run_id: str,
    data_dir: str,
    strategy: Literal["ddp", "fsdp"],
    compile_model: bool,
    attention_backend: str,
    extra_overrides: tuple[str, ...],
) -> list[str]:
    return [
        "train",
        "--config-name",
        config_name,
        f"data.data_dir={data_dir}",
        f"run.run_id={run_id}",
        "training.device=cuda",
        "training.precision=mixed",
        f"training.distributed_strategy={strategy}",
        f"training.compile={str(compile_model).lower()}",
        f"training.attention_backend={attention_backend}",
        *extra_overrides,
    ]


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_DIR, check=True)


@app.function(gpu=DEFAULT_GPU, volumes={DATA_VOLUME_PATH: volume}, secrets=[wandb_secret], timeout=60 * 60 * 24)
def train_single_node(
    run_id: str = "modal-single",
    config_name: str = DEFAULT_CONFIG_NAME,
    data_dir: str = DEFAULT_DATA_DIR,
    strategy: Literal["ddp", "fsdp"] = "ddp",
    compile_model: bool = True,
    attention_backend: str = "flash",
    extra_overrides: tuple[str, ...] = (),
) -> None:
    """Run the existing CLI on one Modal GPU node."""

    _run(
        [
            "uv",
            "run",
            "mega-trading",
            *_training_overrides(config_name, run_id, data_dir, strategy, compile_model, attention_backend, extra_overrides),
        ]
    )
    volume.commit()


@app.function(gpu=DEFAULT_GPU, volumes={DATA_VOLUME_PATH: volume}, secrets=[wandb_secret], timeout=60 * 60 * 24)
@modal.experimental.clustered(size=2, rdma=True)
def train_clustered(
    run_id: str = "modal-cluster",
    config_name: str = DEFAULT_CONFIG_NAME,
    data_dir: str = DEFAULT_DATA_DIR,
    strategy: Literal["ddp", "fsdp"] = "fsdp",
    compile_model: bool = True,
    attention_backend: str = "flash",
    nproc_per_node: int = 8,
    master_port: int = 29500,
    extra_overrides: tuple[str, ...] = (),
) -> None:
    """Run multi-node torchrun on Modal clustered containers."""

    cluster_info = modal.experimental.get_cluster_info()
    _run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "torch.distributed.run",
            f"--nnodes={len(cluster_info.container_ips)}",
            f"--node-rank={cluster_info.rank}",
            f"--master-addr={cluster_info.container_ips[0]}",
            f"--master-port={master_port}",
            f"--nproc-per-node={nproc_per_node}",
            "--module",
            "mega_trading.cli",
            *_training_overrides(config_name, run_id, data_dir, strategy, compile_model, attention_backend, extra_overrides),
        ]
    )
    if cluster_info.rank == 0:
        volume.commit()


@app.local_entrypoint()
def main(
    mode: Literal["single", "cluster"] = "single",
    run_id: str = "modal",
    config_name: str = DEFAULT_CONFIG_NAME,
    data_dir: str = DEFAULT_DATA_DIR,
    strategy: Literal["ddp", "fsdp"] = "ddp",
    compile_model: bool = True,
    attention_backend: str = "flash",
    max_steps: int = 100,
) -> None:
    overrides = (f"training.max_steps={max_steps}",)
    if mode == "cluster":
        train_clustered.remote(
            run_id=run_id,
            config_name=config_name,
            data_dir=data_dir,
            strategy=strategy,
            compile_model=compile_model,
            attention_backend=attention_backend,
            extra_overrides=overrides,
        )
    else:
        train_single_node.remote(
            run_id=run_id,
            config_name=config_name,
            data_dir=data_dir,
            strategy=strategy,
            compile_model=compile_model,
            attention_backend=attention_backend,
            extra_overrides=overrides,
        )
