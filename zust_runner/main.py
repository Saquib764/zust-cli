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

from typing import Any

from zust_runner.http_client import create_runner_client, merge_runner_headers, with_transport_retries
from zust_runner.metrics import run_training_with_metrics
from zust_runner.outbound import RunOutbound
from zust_runner.runner_cache import EFFECTIVE_CONFIG_FILE, EFFECTIVE_ENV_FILE, fetch_cached_bytes, outbound_dir

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
RESTART_EXIT_CODE = 143
PURGE_EXIT_CODE = 144


def _runner_epoch() -> int:
    return int(os.environ.get("ZUST_RUNNER_EPOCH", "1") or "1")


def _warm_restart() -> bool:
    return os.environ.get("ZUST_WARM_RESTART", "").strip() == "1"


def _skip_setup() -> bool:
    return os.environ.get("ZUST_SKIP_SETUP", "").strip() == "1"


def _control_exit(control: int) -> int | None:
    if control == 2:
        return RESTART_EXIT_CODE
    if control == 3:
        return PURGE_EXIT_CODE
    if control:
        return 130
    return None


def _stopped_for_control(code: int) -> bool:
    return code in (RESTART_EXIT_CODE, PURGE_EXIT_CODE)


def _handle_control(outbound: RunOutbound, **fields: Any) -> int | None:
    exit_code = _control_exit(outbound.post_progress(**fields))
    if exit_code is None:
        return None
    if exit_code == 130:
        if outbound._last_run_cancelled:
            outbound.post_progress(
                execution_status="CANCELLED",
                phase="finished",
                failed_stage="cancelled",
            )
        else:
            outbound.post_progress(
                execution_status="FAILED",
                phase="finished",
                failed_stage="stopped",
                cancel_requested=False,
            )
    return exit_code


def resolve_workspace_root() -> Path:
    """Run directory. Local launches set ZUST_WORKSPACE_ROOT; Runpod uses /workspace/runs/<id>."""
    override = os.environ.get("ZUST_WORKSPACE_ROOT", "").strip()
    if override:
        return Path(override)
    return Path("/workspace/runs") / os.environ.get("ZUST_RUN_ID", "")


def _headers() -> dict[str, str]:
    return merge_runner_headers({"Authorization": f"Bearer {RUN_TOKEN}"})


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


def _use_image_torch() -> bool:
    return os.environ.get("ZUST_USE_IMAGE_TORCH", "").strip() == "1"


def _venv_includes_system_site_packages(cwd: Path) -> bool:
    config = cwd / ".venv" / "pyvenv.cfg"
    if not config.is_file():
        return False
    for line in config.read_text().splitlines():
        if line.strip() == "include-system-site-packages = true":
            return True
    return False


def _with_uv_flag(argv: list[str], flag: str, value: str | None = None) -> list[str]:
    if value is None:
        if flag in argv:
            return argv
        return [*argv[:2], flag, *argv[2:]]
    for index, token in enumerate(argv):
        if token == flag and index + 1 < len(argv) and argv[index + 1] == value:
            return argv
        if token == f"{flag}={value}":
            return argv
    return [*argv[:2], flag, value, *argv[2:]]


# PyPI torch wheels pull these in; Runpod PyTorch images already ship CUDA libs on the system interpreter.
_IMAGE_TORCH_SKIP_PACKAGES = (
    "torch",
    "triton",
    "nvidia-cublas-cu12",
    "nvidia-cuda-cupti-cu12",
    "nvidia-cuda-nvrtc-cu12",
    "nvidia-cudnn-cu12",
    "nvidia-cufft-cu12",
    "nvidia-curand-cu12",
    "nvidia-cusolver-cu12",
    "nvidia-cusparse-cu12",
    "nvidia-cusparselt-cu12",
    "nvidia-nccl-cu12",
    "nvidia-nvjitlink-cu12",
    "nvidia-nvtx-cu12",
)


def _with_uv_skip_image_torch_packages(argv: list[str]) -> list[str]:
    result = argv
    for package in _IMAGE_TORCH_SKIP_PACKAGES:
        result = _with_uv_flag(result, "--no-install-package", package)
    return result


def _image_torch_steps(argv: list[str], cwd: Path) -> tuple[list[list[str]], list[str]]:
    """Keep the image's PyTorch instead of downloading the PyPI wheel into .venv.

    uv sync installs torch, and the CUDA nvidia wheels it depends on, into an
    isolated virtualenv. uv run syncs again before the command. On a Runpod
    PyTorch image those packages are already on the system interpreter.
    """
    if not _use_image_torch() or len(argv) < 2 or Path(argv[0]).name != "uv":
        return [], argv
    if argv[1] not in {"sync", "run"}:
        return [], argv
    prelude: list[list[str]] = []
    if not _venv_includes_system_site_packages(cwd):
        prelude.append(
            ["uv", "venv", "--python", sys.executable, "--system-site-packages", ".venv"]
        )
    if argv[1] == "run":
        prelude.append(_with_uv_skip_image_torch_packages(["uv", "sync"]))
        return prelude, _with_uv_flag(argv, "--no-sync")
    return prelude, _with_uv_skip_image_torch_packages(argv)


def _run_argv(outbound: RunOutbound, argv: list[str], cwd: Path) -> int:
    if not argv:
        return 0
    prelude, argv = _image_torch_steps(argv, cwd)
    for step in prelude:
        code = _run_argv(outbound, step, cwd)
        if code != 0:
            return code
    env = None
    if _use_image_torch():
        env = os.environ.copy()
        env["UV_PYTHON_PREFERENCE"] = "only-system"
    process = subprocess.Popen(
        argv,
        cwd=str(cwd),
        env=env,
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
        exit_code = _control_exit(outbound.post_progress())
        if exit_code is not None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            stdout_thread.join()
            stderr_thread.join()
            return exit_code
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


def _maybe_clean_tensorboard(outbound: RunOutbound, checkout: Path) -> None:
    if os.environ.get("ZUST_CLEAN_TENSORBOARD") != "1":
        return
    from zust_runner.tensorboard_clean import clean_tensorboard_events

    removed = clean_tensorboard_events(checkout, CONFIG_PATH)
    outbound.append_log(
        "stdout",
        f"Cleared {removed} TensorBoard event file(s) before training restart.",
    )
    outbound.post_progress(runner_clean_tensorboard=False)


def _clone_commit(checkout: Path, outbound: RunOutbound) -> None:
    if _warm_restart() and (checkout / ".git").is_dir():
        return
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
        with create_runner_client() as client:
            outbound = RunOutbound(
                client,
                api_base=API_BASE,
                run_id=RUN_ID,
                headers=_headers(),
                cache_dir=cache_dir,
                runner_epoch=_runner_epoch(),
            )
            stop_heartbeat = threading.Event()

            def _heartbeat() -> None:
                from zust_runner.workspace_snapshot import workspace_snapshot_payload

                while not stop_heartbeat.wait(30):
                    outbound.post_progress()
                    try:
                        outbound.post_workspace_snapshot(workspace_snapshot_payload())
                    except Exception:
                        pass

            threading.Thread(target=_heartbeat, daemon=True).start()
            try:
                return _execute(client, outbound, checkout)
            finally:
                stop_heartbeat.set()
                outbound.flush()
    except Exception as exc:
        print(f"Runner failed: {exc}", file=sys.stderr)
        try:
            with create_runner_client() as client:
                failed_outbound = RunOutbound(
                    client,
                    api_base=API_BASE,
                    run_id=RUN_ID,
                    headers=_headers(),
                    cache_dir=cache_dir,
                    runner_epoch=_runner_epoch(),
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

    interrupted = _handle_control(outbound, phase="cloning", execution_status="RUNNING")
    if interrupted is not None:
        return interrupted

    _clone_commit(checkout, outbound)
    _maybe_clean_tensorboard(outbound, checkout)
    try:
        from zust_runner.workspace_snapshot import workspace_snapshot_payload

        outbound.post_workspace_snapshot(workspace_snapshot_payload())
    except Exception:
        pass

    if SETUP_ARGV and not _skip_setup():
        interrupted = _handle_control(outbound, phase="setup")
        if interrupted is not None:
            return interrupted
        setup_code = _run_argv(outbound, SETUP_ARGV, checkout)
        if _stopped_for_control(setup_code):
            return setup_code
        if setup_code != 0:
            failed_stage = "setup"
            outbound.post_progress(
                execution_status="FAILED",
                phase="finished",
                failed_stage=failed_stage,
                setup_exit_code=setup_code,
            )
            return setup_code

    interrupted = _handle_control(outbound, phase="writing_config")
    if interrupted is not None:
        return interrupted

    cache_root = outbound_dir()
    config_bytes = fetch_cached_bytes(
        client,
        url=f"{API_BASE}/api/v1/internal/runs/{RUN_ID}/effective-config",
        headers=_headers(),
        cache_path=cache_root / EFFECTIVE_CONFIG_FILE,
    )
    env_bytes = fetch_cached_bytes(
        client,
        url=f"{API_BASE}/api/v1/internal/runs/{RUN_ID}/effective-env",
        headers=_headers(),
        cache_path=cache_root / EFFECTIVE_ENV_FILE,
    )
    _write_config_bytes(checkout, CONFIG_PATH, config_bytes)
    _write_repo_dotenv(checkout, env_bytes)

    interrupted = _handle_control(outbound, phase="training")
    if interrupted is not None:
        return interrupted
    training_code = run_training_with_metrics(
        client,
        TRAINING_ARGV,
        checkout,
        CONFIG_PATH,
        run_argv=lambda _client, argv, cwd: _run_argv(outbound, argv, cwd),
        post_metrics=outbound.post_metrics,
    )
    if _stopped_for_control(training_code):
        return training_code
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
        interrupted = _handle_control(outbound, phase="inference")
        if interrupted is not None:
            return interrupted
        inference_code = _run_argv(outbound, INFERENCE_ARGV, checkout)
        if _stopped_for_control(inference_code):
            return inference_code
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
