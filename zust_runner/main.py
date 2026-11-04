"""Execute a single training run inside a Runpod workspace."""

from __future__ import annotations

import codecs
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import httpx

from zust_runner.metrics import run_training_with_metrics
from zust_runner.outbound import RunOutbound

API_BASE = os.environ.get("ZUST_API_BASE", "").rstrip("/")
RUN_ID = os.environ.get("ZUST_RUN_ID", "")
RUN_TOKEN = os.environ.get("ZUST_RUN_TOKEN", "")
REPOSITORY = os.environ.get("ZUST_REPOSITORY", "")
COMMIT_SHA = os.environ.get("ZUST_COMMIT_SHA", "")
GIT_BRANCH = (os.environ.get("ZUST_GIT_BRANCH", "main") or "main").strip() or "main"
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


class StreamSplitter:
    """Turn a process byte stream into log records.

    A newline ends a record. A carriage return, which is how progress bars
    redraw the current row, is its own record prefixed with ``\\r`` so the
    viewer can overwrite that row instead of appending another line.
    """

    def __init__(self) -> None:
        self.buf = ""
        self.overwrite = False
        self.published = True
        self.held_cr = False
        self.last_cr: str | None = None

    def feed(self, data: str) -> list[str]:
        out: list[str] = []
        if self.held_cr:
            self.held_cr = False
            if data.startswith("\n"):
                out.extend(self._newline())
                data = data[1:]
            else:
                self._enter_cr(out)
        if data.endswith("\r") and not data.endswith("\r\n"):
            self.held_cr = True
            data = data[:-1]
        self._consume(data, out)
        if self.held_cr:
            return out
        if self.overwrite and self.buf and not self.published:
            self._publish_cr(out)
        return out

    def finish(self) -> list[str]:
        out: list[str] = []
        if self.held_cr:
            self.held_cr = False
            self._enter_cr(out)
        if self.buf and not self.published:
            if self.overwrite:
                self._publish_cr(out)
            else:
                out.append(self.buf)
                self.buf = ""
                self.published = True
        return out

    def _consume(self, data: str, out: list[str]) -> None:
        index = 0
        while index < len(data):
            char = data[index]
            if char == "\r" and index + 1 < len(data) and data[index + 1] == "\n":
                out.extend(self._newline())
                index += 2
                continue
            if char == "\r":
                self._enter_cr(out)
                index += 1
                continue
            if char == "\n":
                out.extend(self._newline())
                index += 1
                continue
            self.buf += char
            self.published = False
            index += 1

    def _enter_cr(self, out: list[str]) -> None:
        if self.buf and not self.published:
            if self.overwrite:
                self._publish_cr(out)
            else:
                out.append(self.buf)
                self.last_cr = None
        self.overwrite = True
        self.buf = ""
        self.published = True

    def _newline(self) -> list[str]:
        out: list[str] = []
        if self.overwrite:
            if self.buf and not self.published:
                self._publish_cr(out)
            out.append("\n")
        else:
            out.append(self.buf)
        self.buf = ""
        self.overwrite = False
        self.published = True
        self.last_cr = None
        return out

    def _publish_cr(self, out: list[str]) -> None:
        self.published = True
        if not self.buf.strip():
            return
        record = "\r" + self.buf
        if record == self.last_cr:
            return
        self.last_cr = record
        out.append(record)


def _drain(outbound: RunOutbound, stream: str, pipe) -> None:
    splitter = StreamSplitter()
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    while True:
        chunk = pipe.read(4096)
        if not chunk:
            break
        for record in splitter.feed(decoder.decode(chunk)):
            with _LOG_LOCK:
                outbound.append_log(stream, record)
    for record in splitter.feed(decoder.decode(b"", final=True)):
        with _LOG_LOCK:
            outbound.append_log(stream, record)
    for record in splitter.finish():
        with _LOG_LOCK:
            outbound.append_log(stream, record)


def _run_argv(outbound: RunOutbound, argv: list[str], cwd: Path) -> int:
    if not argv:
        return 0
    process = subprocess.Popen(
        argv,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        shell=False,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    stdout_thread = threading.Thread(target=_drain, args=(outbound, "stdout", process.stdout), daemon=True)
    stderr_thread = threading.Thread(target=_drain, args=(outbound, "stderr", process.stderr), daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    while process.poll() is None:
        if outbound.post_progress():
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            stdout_thread.join()
            stderr_thread.join()
            return 130
        time.sleep(1.0)
    code = process.returncode if process.returncode is not None else process.wait()
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


def _write_bytes_atomic(destination: Path, content: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=destination.parent) as handle:
        handle.write(content)
        temp_name = handle.name
    os.replace(temp_name, destination)


def _write_config_bytes(checkout: Path, config_path: str, content: bytes) -> None:
    destination = _validate_repo_relative_path(checkout, config_path)
    _write_bytes_atomic(destination, content)


def _write_repo_dotenv(checkout: Path, content: bytes) -> None:
    if not content.strip():
        return
    _write_bytes_atomic(checkout / ".env", content)


def _github_remote_url(owner: str, repo: str) -> str:
    token = os.environ.get("ZUST_GITHUB_TOKEN", "").strip()
    if token:
        return f"https://x-access-token:{token}@github.com/{owner}/{repo}.git"
    return f"https://github.com/{owner}/{repo}.git"


def _clone_commit(checkout: Path, outbound: RunOutbound) -> None:
    owner, repo = REPOSITORY.split("/", 1)
    remote = _github_remote_url(owner, repo)
    checkout.mkdir(parents=True, exist_ok=True)
    _run_argv(outbound, ["git", "init", "-b", GIT_BRANCH], checkout)
    _run_argv(outbound, ["git", "remote", "add", "origin", remote], checkout)
    _run_argv(outbound, ["git", "fetch", "--depth", "1", "origin", COMMIT_SHA], checkout)
    _run_argv(outbound, ["git", "checkout", "--detach", "FETCH_HEAD"], checkout)


def run() -> int:
    if not (API_BASE and RUN_ID and RUN_TOKEN and REPOSITORY and COMMIT_SHA and CONFIG_PATH):
        print("Missing required runner environment", file=sys.stderr)
        return 2

    checkout = resolve_workspace_root() / "repo"
    cache_dir = resolve_workspace_root() / "outbound"
    try:
        with httpx.Client() as client:
            outbound = RunOutbound(
                client,
                api_base=API_BASE,
                run_id=RUN_ID,
                headers=_headers(),
                cache_dir=cache_dir,
            )
            stop_heartbeat = threading.Event()

            def _heartbeat() -> None:
                while not stop_heartbeat.wait(30):
                    outbound.post_progress()

            threading.Thread(target=_heartbeat, daemon=True).start()
            try:
                return _execute(client, outbound, checkout)
            finally:
                stop_heartbeat.set()
                outbound.flush()
    except Exception as exc:
        print(f"Runner failed: {exc}", file=sys.stderr)
        try:
            with httpx.Client() as client:
                failed_outbound = RunOutbound(
                    client,
                    api_base=API_BASE,
                    run_id=RUN_ID,
                    headers=_headers(),
                    cache_dir=cache_dir,
                )
                failed_outbound.post_progress(
                    execution_status="FAILED",
                    phase="finished",
                    failed_stage="runner",
                )
                failed_outbound.flush()
        except Exception:
            pass
        return 1


def _execute(client: httpx.Client, outbound: RunOutbound, checkout: Path) -> int:
    failed_stage: str | None = None
    setup_code: int | None = None
    training_code: int | None = None
    inference_code: int | None = None
    execution_status = "FAILED"

    if outbound.post_progress(phase="cloning", execution_status="RUNNING"):
        outbound.post_progress(execution_status="CANCELLED", phase="finished", failed_stage="cancelled")
        return 130

    _clone_commit(checkout, outbound)

    if SETUP_ARGV:
        if outbound.post_progress(phase="setup"):
            outbound.post_progress(execution_status="CANCELLED", phase="finished", failed_stage="cancelled")
            return 130
        setup_code = _run_argv(outbound, SETUP_ARGV, checkout)
        if setup_code != 0:
            failed_stage = "setup"
            outbound.post_progress(
                execution_status="FAILED",
                phase="finished",
                failed_stage=failed_stage,
                setup_exit_code=setup_code,
            )
            return setup_code

    if outbound.post_progress(phase="writing_config"):
        outbound.post_progress(execution_status="CANCELLED", phase="finished", failed_stage="cancelled")
        return 130

    config_response = client.get(
        f"{API_BASE}/api/v1/internal/runs/{RUN_ID}/effective-config",
        headers=_headers(),
        timeout=30.0,
    )
    config_response.raise_for_status()
    _write_config_bytes(checkout, CONFIG_PATH, config_response.content)

    env_response = client.get(
        f"{API_BASE}/api/v1/internal/runs/{RUN_ID}/effective-env",
        headers=_headers(),
        timeout=30.0,
    )
    env_response.raise_for_status()
    _write_repo_dotenv(checkout, env_response.content)

    if outbound.post_progress(phase="training"):
        outbound.post_progress(execution_status="CANCELLED", phase="finished", failed_stage="cancelled")
        return 130
    training_code = run_training_with_metrics(
        client,
        TRAINING_ARGV,
        checkout,
        CONFIG_PATH,
        run_argv=lambda _client, argv, cwd: _run_argv(outbound, argv, cwd),
        post_metrics=outbound.post_metrics,
    )
    if training_code != 0:
        failed_stage = "training"
        outbound.post_progress(
            execution_status="FAILED",
            phase="finished",
            failed_stage=failed_stage,
            setup_exit_code=setup_code,
            training_exit_code=training_code,
        )
        return training_code

    execution_status = "COMPLETED"
    if RUN_TEST_INFERENCE and INFERENCE_ARGV:
        if outbound.post_progress(phase="inference"):
            outbound.post_progress(execution_status="CANCELLED", phase="finished", failed_stage="cancelled")
            return 130
        inference_code = _run_argv(outbound, INFERENCE_ARGV, checkout)
        if inference_code != 0:
            failed_stage = "inference"
            execution_status = "FAILED"
            outbound.post_progress(
                execution_status=execution_status,
                phase="finished",
                failed_stage=failed_stage,
                setup_exit_code=setup_code,
                training_exit_code=training_code,
                inference_exit_code=inference_code,
            )
            return inference_code

    outbound.post_progress(
        execution_status=execution_status,
        phase="finished",
        setup_exit_code=setup_code,
        training_exit_code=training_code,
        inference_exit_code=inference_code,
    )
    return 0
