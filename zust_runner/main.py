"""Execute a single training run inside a Runpod workspace."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import httpx

API_BASE = os.environ.get("ZUST_API_BASE", "").rstrip("/")
RUN_ID = os.environ.get("ZUST_RUN_ID", "")
RUN_TOKEN = os.environ.get("ZUST_RUN_TOKEN", "")
REPOSITORY = os.environ.get("ZUST_REPOSITORY", "")
COMMIT_SHA = os.environ.get("ZUST_COMMIT_SHA", "")
CONFIG_PATH = os.environ.get("ZUST_CONFIG_PATH", "")
RUN_TEST_INFERENCE = os.environ.get("ZUST_RUN_TEST_INFERENCE", "0") == "1"
SETUP_ARGV = json.loads(os.environ.get("ZUST_SETUP_ARGV", "[]"))
TRAINING_ARGV = json.loads(os.environ.get("ZUST_TRAINING_ARGV", "[]"))
INFERENCE_ARGV = json.loads(os.environ.get("ZUST_INFERENCE_ARGV", "[]"))

_LOG_LOCK = threading.Lock()


def resolve_workspace_root() -> Path:
    """Run directory. Local launches set ZUST_WORKSPACE_ROOT; Runpod uses /workspace/runs/<id>."""
    override = os.environ.get("ZUST_WORKSPACE_ROOT", "").strip()
    if override:
        return Path(override)
    return Path("/workspace/runs") / os.environ.get("ZUST_RUN_ID", "")


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {RUN_TOKEN}"}


def _with_retries(action):
    delay = 1.0
    last_error: Exception | None = None
    for attempt in range(8):
        try:
            return action()
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            last_error = exc
            if attempt == 7:
                raise
            time.sleep(delay)
            delay = min(delay * 1.5, 8.0)
    if last_error is not None:
        raise last_error
    raise RuntimeError("retry loop exited unexpectedly")


def _post_progress(client: httpx.Client, **fields) -> bool:
    def _send() -> bool:
        response = client.post(
            f"{API_BASE}/api/v1/internal/runs/{RUN_ID}/progress",
            headers=_headers(),
            json=fields,
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        return bool(data.get("cancel"))

    return _with_retries(_send)


def _append_log(client: httpx.Client, stream: str, line: str) -> None:
    def _send() -> None:
        client.post(
            f"{API_BASE}/api/v1/internal/runs/{RUN_ID}/logs",
            headers=_headers(),
            json={"entries": [{"stream": stream, "line": line}]},
            timeout=30.0,
        ).raise_for_status()

    with _LOG_LOCK:
        _with_retries(_send)


def _drain(client: httpx.Client, stream: str, pipe) -> None:
    for line in pipe:
        _append_log(client, stream, line.rstrip("\n"))


def _run_argv(client: httpx.Client, argv: list[str], cwd: Path) -> int:
    if not argv:
        return 0
    process = subprocess.Popen(
        argv,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    stdout_thread = threading.Thread(target=_drain, args=(client, "stdout", process.stdout), daemon=True)
    stderr_thread = threading.Thread(target=_drain, args=(client, "stderr", process.stderr), daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    code = process.wait()
    stdout_thread.join()
    stderr_thread.join()
    return code


def _validate_repo_relative_path(checkout: Path, config_path: str) -> Path:
    normalized = config_path.replace("\\", "/").lstrip("/")
    if ".." in normalized.split("/"):
        raise ValueError("Config path escapes repository root")
    destination = (checkout / normalized).resolve()
    checkout_resolved = checkout.resolve()
    if destination != checkout_resolved and checkout_resolved not in destination.parents:
        raise ValueError("Config path escapes repository root")
    return destination


def _write_config_bytes(checkout: Path, config_path: str, content: bytes) -> None:
    destination = _validate_repo_relative_path(checkout, config_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=destination.parent) as handle:
        handle.write(content)
        temp_name = handle.name
    os.replace(temp_name, destination)


def _clone_commit(checkout: Path, client: httpx.Client) -> None:
    owner, repo = REPOSITORY.split("/", 1)
    remote = f"https://github.com/{owner}/{repo}.git"
    checkout.mkdir(parents=True, exist_ok=True)
    _run_argv(client, ["git", "init"], checkout)
    _run_argv(client, ["git", "remote", "add", "origin", remote], checkout)
    _run_argv(client, ["git", "fetch", "--depth", "1", "origin", COMMIT_SHA], checkout)
    _run_argv(client, ["git", "checkout", "--detach", "FETCH_HEAD"], checkout)


def run() -> int:
    if not (API_BASE and RUN_ID and RUN_TOKEN and REPOSITORY and COMMIT_SHA and CONFIG_PATH):
        print("Missing required runner environment", file=sys.stderr)
        return 2

    checkout = resolve_workspace_root() / "repo"
    try:
        with httpx.Client() as client:
            stop_heartbeat = threading.Event()

            def _heartbeat() -> None:
                while not stop_heartbeat.wait(30):
                    try:
                        _with_retries(
                            lambda: client.post(
                                f"{API_BASE}/api/v1/internal/runs/{RUN_ID}/progress",
                                headers=_headers(),
                                json={},
                                timeout=30.0,
                            ).raise_for_status()
                        )
                    except Exception:
                        continue

            threading.Thread(target=_heartbeat, daemon=True).start()
            try:
                return _execute(client, checkout)
            finally:
                stop_heartbeat.set()
    except Exception as exc:
        print(f"Runner failed: {exc}", file=sys.stderr)
        try:
            with httpx.Client() as client:
                _post_progress(
                    client,
                    execution_status="FAILED",
                    phase="finished",
                    failed_stage="runner",
                )
        except Exception:
            pass
        return 1


def _execute(client: httpx.Client, checkout: Path) -> int:
    failed_stage: str | None = None
    setup_code: int | None = None
    training_code: int | None = None
    inference_code: int | None = None
    execution_status = "FAILED"

    if _post_progress(client, phase="cloning", execution_status="RUNNING"):
        _post_progress(client, execution_status="CANCELLED", phase="finished", failed_stage="cancelled")
        return 130

    _clone_commit(checkout, client)

    if SETUP_ARGV:
        if _post_progress(client, phase="setup"):
            _post_progress(client, execution_status="CANCELLED", phase="finished", failed_stage="cancelled")
            return 130
        setup_code = _run_argv(client, SETUP_ARGV, checkout)
        if setup_code != 0:
            failed_stage = "setup"
            _post_progress(
                client,
                execution_status="FAILED",
                phase="finished",
                failed_stage=failed_stage,
                setup_exit_code=setup_code,
            )
            return setup_code

    if _post_progress(client, phase="writing_config"):
        _post_progress(client, execution_status="CANCELLED", phase="finished", failed_stage="cancelled")
        return 130

    config_response = client.get(
        f"{API_BASE}/api/v1/internal/runs/{RUN_ID}/effective-config",
        headers=_headers(),
        timeout=30.0,
    )
    config_response.raise_for_status()
    _write_config_bytes(checkout, CONFIG_PATH, config_response.content)

    if _post_progress(client, phase="training"):
        _post_progress(client, execution_status="CANCELLED", phase="finished", failed_stage="cancelled")
        return 130
    training_code = _run_argv(client, TRAINING_ARGV, checkout)
    if training_code != 0:
        failed_stage = "training"
        _post_progress(
            client,
            execution_status="FAILED",
            phase="finished",
            failed_stage=failed_stage,
            setup_exit_code=setup_code,
            training_exit_code=training_code,
        )
        return training_code

    execution_status = "COMPLETED"
    if RUN_TEST_INFERENCE and INFERENCE_ARGV:
        if _post_progress(client, phase="inference"):
            _post_progress(client, execution_status="CANCELLED", phase="finished", failed_stage="cancelled")
            return 130
        inference_code = _run_argv(client, INFERENCE_ARGV, checkout)
        if inference_code != 0:
            failed_stage = "inference"
            execution_status = "FAILED"
            _post_progress(
                client,
                execution_status=execution_status,
                phase="finished",
                failed_stage=failed_stage,
                setup_exit_code=setup_code,
                training_exit_code=training_code,
                inference_exit_code=inference_code,
            )
            return inference_code

    _post_progress(
        client,
        execution_status=execution_status,
        phase="finished",
        setup_exit_code=setup_code,
        training_exit_code=training_code,
        inference_exit_code=inference_code,
    )
    return 0
