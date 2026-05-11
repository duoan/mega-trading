"""Start or reuse a local MLflow tracking server for Makefile workflows."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import NamedTuple


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000
DEFAULT_BACKEND_STORE_URI = "sqlite:///.mega-trading/mlflow/mlflow.db"
DEFAULT_ARTIFACT_ROOT = Path(".mega-trading/mlflow/artifacts")
DEFAULT_PID_FILE = Path(".mega-trading/mlflow/mlflow-server.pid")
DEFAULT_LOG_FILE = Path(".mega-trading/mlflow/mlflow-server.log")
DEFAULT_TIMEOUT_SECONDS = 30.0


class EnsureOptions(NamedTuple):
    host: str
    port: int
    backend_store_uri: str
    default_artifact_root: Path
    pid_file: Path
    log_file: Path
    timeout_seconds: float


def ensure_server(options: EnsureOptions) -> None:
    """Ensure an MLflow server is reachable, starting one only when needed."""
    if _server_ready(options.host, options.port):
        print(f"MLflow server already running at {_tracking_uri(options.host, options.port)}")
        return

    _prepare_storage(options)
    command = _mlflow_server_command(options)
    with options.log_file.open("ab") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

    options.pid_file.write_text(f"{process.pid}\n", encoding="utf-8")
    if not _wait_for_server(options):
        raise RuntimeError(
            "MLflow server did not become ready within "
            f"{options.timeout_seconds:g}s; see {options.log_file} for details"
        )
    print(f"Started MLflow server at {_tracking_uri(options.host, options.port)}")
    print(f"MLflow log: {options.log_file}")


def stop_server(pid_file: Path) -> None:
    if not pid_file.exists():
        print(f"No MLflow pid file found at {pid_file}")
        return

    pid_text = pid_file.read_text(encoding="utf-8").strip()
    try:
        pid = int(pid_text)
    except ValueError as exc:
        raise RuntimeError(f"Invalid MLflow pid file at {pid_file}: {pid_text!r}") from exc

    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Stopped MLflow server process {pid}")
    except ProcessLookupError:
        print(f"MLflow server process {pid} is not running")
    finally:
        pid_file.unlink(missing_ok=True)


def _tracking_uri(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def _server_ready(host: str, port: int, timeout_seconds: float = 1.0) -> bool:
    request = urllib.request.Request(f"{_tracking_uri(host, port)}/health")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return 200 <= response.status < 500
    except (OSError, TimeoutError, urllib.error.URLError):
        return False


def _wait_for_server(options: EnsureOptions) -> bool:
    deadline = time.monotonic() + options.timeout_seconds
    while time.monotonic() < deadline:
        if _server_ready(options.host, options.port):
            return True
        time.sleep(0.5)
    return _server_ready(options.host, options.port)


def _prepare_storage(options: EnsureOptions) -> None:
    options.default_artifact_root.mkdir(parents=True, exist_ok=True)
    options.pid_file.parent.mkdir(parents=True, exist_ok=True)
    options.log_file.parent.mkdir(parents=True, exist_ok=True)
    sqlite_path = _sqlite_path(options.backend_store_uri)
    if sqlite_path is not None:
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)


def _sqlite_path(backend_store_uri: str) -> Path | None:
    prefix = "sqlite:///"
    if not backend_store_uri.startswith(prefix):
        return None
    return Path(backend_store_uri.removeprefix(prefix)).expanduser()


def _mlflow_server_command(options: EnsureOptions) -> list[str]:
    return [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--host",
        options.host,
        "--port",
        str(options.port),
        "--backend-store-uri",
        options.backend_store_uri,
        "--default-artifact-root",
        str(options.default_artifact_root),
        "--serve-artifacts",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Start or stop the local Mega-Trading MLflow server.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--backend-store-uri", default=DEFAULT_BACKEND_STORE_URI)
    parser.add_argument("--default-artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--pid-file", type=Path, default=DEFAULT_PID_FILE)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--stop", action="store_true", help="stop the server recorded in the pid file")
    args = parser.parse_args()

    options = EnsureOptions(
        host=args.host,
        port=args.port,
        backend_store_uri=args.backend_store_uri,
        default_artifact_root=args.default_artifact_root,
        pid_file=args.pid_file,
        log_file=args.log_file,
        timeout_seconds=args.timeout_seconds,
    )
    if args.stop:
        stop_server(options.pid_file)
    else:
        ensure_server(options)
        print(f"Set training.mlflow_tracking_uri={_tracking_uri(options.host, options.port)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
