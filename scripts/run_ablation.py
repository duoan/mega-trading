"""Expand and run model-size/data-size ablation grids."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Mega-Trading ablation grid.")
    parser.add_argument("config", help="path to ablation YAML")
    parser.add_argument("--mlflow-tracking-uri", help="MLflow tracking server URI for all stages")
    parser.add_argument("--dry-run", action="store_true", help="print commands without executing them")
    args = parser.parse_args()

    commands = plan_commands(Path(args.config), mlflow_tracking_uri=args.mlflow_tracking_uri)
    for command in commands:
        print("+ " + " ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)
    return 0


def plan_commands(config_path: Path, mlflow_tracking_uri: str | None = None) -> list[list[str]]:
    """Return the command plan for a data-size x model-size ablation grid."""
    config = _load_config(config_path)
    name = _required_string(config, "name")
    config_name = str(config.get("config_name", "default"))
    ingest_config = _required_string(config, "ingest_config")
    data_dir = _required_string(config, "data_dir")
    base_overrides = _strings(config.get("base_overrides", []))
    data_sizes = _named_items(config, "data_sizes")
    model_sizes = _named_items(config, "model_sizes")
    backtest_batches = int(dict(config.get("backtest", {})).get("max_batches", 128))
    examples_config = dict(config.get("examples", {}))
    max_sequences = int(examples_config.get("max_sequences", 3))
    max_tokens = int(examples_config.get("max_tokens", 16))

    commands: list[list[str]] = []
    for data_size in data_sizes:
        data_name = str(data_size["name"])
        data_overrides = _data_overrides(name, data_name, data_dir, _strings(data_size.get("overrides", [])))
        commands.append(
            [
                "uv",
                "run",
                "python",
                "scripts/prepare_numpy_dataset.py",
                "--ingest-config",
                ingest_config,
                "--config-name",
                config_name,
                *base_overrides,
                *data_overrides,
            ]
        )
        for model_size in model_sizes:
            model_name = str(model_size["name"])
            model_overrides = _strings(model_size.get("overrides", []))
            run_id = _run_id(name, data_name, model_name)
            common = [
                "--config-name",
                config_name,
                *base_overrides,
                *data_overrides,
                *model_overrides,
                f"run.run_id={run_id}",
                *_mlflow_overrides(mlflow_tracking_uri),
                f"+ablation.name={name}",
                f"+ablation.data_size={data_name}",
                f"+ablation.model_size={model_name}",
            ]
            commands.extend(
                [
                    ["uv", "run", "mega-trading", "train", *common],
                    [
                        "uv",
                        "run",
                        "mega-trading",
                        "backtest",
                        "--max-batches",
                        str(backtest_batches),
                        *common,
                    ],
                    [
                        "uv",
                        "run",
                        "mega-trading",
                        "backtest-examples",
                        "--max-sequences",
                        str(max_sequences),
                        "--max-tokens",
                        str(max_tokens),
                        *common,
                    ],
                    ["uv", "run", "mega-trading", "report", *common],
                ]
            )
    return commands


def _load_config(config_path: Path) -> dict[str, Any]:
    value = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(value, dict):
        raise ValueError("ablation config must be a mapping")
    return value


def _required_string(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not value:
        raise ValueError(f"ablation config requires {key}")
    return str(value)


def _named_items(config: dict[str, Any], key: str) -> list[dict[str, Any]]:
    values = config.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"ablation config requires non-empty {key}")
    items = []
    for value in values:
        if not isinstance(value, dict) or not value.get("name"):
            raise ValueError(f"each {key} entry requires a name")
        items.append(value)
    return items


def _data_overrides(name: str, data_name: str, data_dir: str, overrides: list[str]) -> list[str]:
    values = [f"data.data_dir={data_dir}"]
    if not any(value.startswith("data.mixture=") for value in overrides):
        values.append(f"data.mixture={_slug(name)}-{_slug(data_name)}")
    values.extend(overrides)
    return values


def _strings(values: object) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, list):
        raise ValueError("overrides must be a list")
    return [str(value) for value in values]


def _mlflow_overrides(tracking_uri: str | None) -> list[str]:
    if not tracking_uri:
        return []
    return [f"training.mlflow_tracking_uri={tracking_uri}"]


def _run_id(name: str, data_name: str, model_name: str) -> str:
    return f"{_slug(name)}__data_{_slug(data_name)}__model_{_slug(model_name)}"


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value.strip().lower()).strip("_")


if __name__ == "__main__":
    raise SystemExit(main())
