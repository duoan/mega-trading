"""Sync the locally logged-in W&B API key into a Modal Secret."""

from __future__ import annotations

import argparse
import json
import netrc
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Create or update a Modal Secret from the local W&B login.")
    parser.add_argument("--secret-name", default="wandb-secret", help="Modal Secret name to create or update")
    args = parser.parse_args()

    api_key = _wandb_api_key()
    modal = shutil.which("modal")
    if modal is None:
        raise RuntimeError("modal CLI was not found; run this script with `uv run python ...` after installing dependencies")

    secret_path = _write_secret_json(api_key)
    try:
        result = subprocess.run(
            [
                modal,
                "secret",
                "create",
                "--force",
                "--from-json",
                str(secret_path),
                args.secret_name,
            ],
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError("could not sync Modal Secret; run `uv run modal token new` and retry")
    finally:
        secret_path.unlink(missing_ok=True)
    print(f"synced W&B API key to Modal Secret `{args.secret_name}`")
    return 0


def _wandb_api_key() -> str:
    env_key = os.environ.get("WANDB_API_KEY")
    if env_key:
        return env_key
    netrc_path = Path.home() / ".netrc"
    if netrc_path.exists():
        try:
            auth = netrc.netrc(str(netrc_path)).authenticators("api.wandb.ai")
        except netrc.NetrcParseError as exc:
            raise RuntimeError(f"could not parse {netrc_path}") from exc
        if auth and auth[2]:
            return auth[2]
    raise RuntimeError("no W&B API key found; run `wandb login` or set WANDB_API_KEY")


def _write_secret_json(api_key: str) -> Path:
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False)
    path = Path(handle.name)
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        json.dump({"WANDB_API_KEY": api_key}, handle)
        handle.write("\n")
    finally:
        handle.close()
    return path


if __name__ == "__main__":
    raise SystemExit(main())
