"""Install flash-attn only for CUDA-capable training environments."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys


DEFAULT_PACKAGE = "flash-attn"


def main() -> int:
    parser = argparse.ArgumentParser(description="Install flash-attn when CUDA is available.")
    parser.add_argument("--package", default=DEFAULT_PACKAGE, help="pip package spec to install")
    parser.add_argument("--require-cuda", action="store_true", help="fail instead of skipping when CUDA is unavailable")
    parser.add_argument("--force-cuda", action="store_true", help="install even when CUDA cannot be detected at build time")
    parser.add_argument("--dry-run", action="store_true", help="print the install command without running it")
    args = parser.parse_args()

    if _flash_attn_installed():
        print("flash-attn already installed; skipping")
        return 0

    if not _cuda_runtime_available(force_cuda=args.force_cuda):
        message = "CUDA not detected; skipping flash-attn install"
        if args.require_cuda:
            raise RuntimeError(message)
        print(message)
        return 0

    command = _install_command(args.package)
    if args.dry_run:
        print("+ " + " ".join(command))
        return 0
    subprocess.run(command, check=True)
    return 0


def _flash_attn_installed() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def _cuda_runtime_available(force_cuda: bool = False) -> bool:
    if force_cuda:
        return True
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch must be installed before installing flash-attn") from exc
    if torch.cuda.is_available():
        return True
    if shutil.which("nvcc") is not None:
        return True
    return bool(os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"))


def _install_command(package: str) -> list[str]:
    uv = shutil.which("uv")
    if uv is not None:
        return [
            uv,
            "pip",
            "install",
            package,
            "--no-build-isolation",
        ]
    return [
        sys.executable,
        "-m",
        "pip",
        "install",
        package,
        "--no-build-isolation",
    ]


if __name__ == "__main__":
    raise SystemExit(main())
