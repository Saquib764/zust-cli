"""Keep the Runpod container alive and restart training without reinstalling zust-cli."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from zust_runner.huggingface_push import run_huggingface_push
from zust_runner.http_client import create_runner_client, merge_runner_headers, with_transport_retries
from zust_runner.main import PURGE_EXIT_CODE, RESTART_EXIT_CODE, resolve_workspace_root

POLL_SECONDS = 3.0


def _headers() -> dict[str, str]:
    token = os.environ.get("ZUST_RUN_TOKEN", "")
    return merge_runner_headers({"Authorization": f"Bearer {token}"})


def _api_base() -> str:
    return os.environ.get("ZUST_API_BASE", "").rstrip("/")


def _run_id() -> str:
    return os.environ.get("ZUST_RUN_ID", "")


def fetch_session(client: httpx.Client) -> dict[str, Any]:
    def _get() -> dict[str, Any]:
        response = client.get(
            f"{_api_base()}/api/v1/internal/runs/{_run_id()}/session",
            headers=_headers(),
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()

    return with_transport_retries(_get)


def complete_action(client: httpx.Client) -> None:
    def _post() -> None:
        client.post(
            f"{_api_base()}/api/v1/internal/runs/{_run_id()}/action-complete",
            headers=_headers(),
            timeout=60.0,
        ).raise_for_status()

    with_transport_retries(_post)


def purge_training_workspace() -> None:
    root = resolve_workspace_root()
    for name in ("repo", "outbound"):
        target = root / name
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)


def _pending_type(session: dict[str, Any]) -> str | None:
    action = session.get("pending_action")
    if isinstance(action, dict) and isinstance(action.get("type"), str):
        return action["type"]
    return None


def _finish_purge(client: httpx.Client) -> None:
    purge_training_workspace()
    complete_action(client)


def _maybe_run_pending_action(client: httpx.Client, session: dict[str, Any]) -> None:
    action = session.get("pending_action")
    if not isinstance(action, dict) or action.get("type") != "huggingface_push":
        return
    checkout = resolve_workspace_root() / "repo"
    run_huggingface_push(action, checkout)
    complete_action(client)


def _child_env(session: dict[str, Any], epoch: int) -> dict[str, str]:
    env = os.environ.copy()
    env["ZUST_RUNNER_EPOCH"] = str(epoch)
    if epoch > 1:
        env["ZUST_WARM_RESTART"] = "1"
    if session.get("skip_setup"):
        env["ZUST_SKIP_SETUP"] = "1"
    else:
        env.pop("ZUST_SKIP_SETUP", None)
    if session.get("clean_tensorboard"):
        env["ZUST_CLEAN_TENSORBOARD"] = "1"
    else:
        env.pop("ZUST_CLEAN_TENSORBOARD", None)
    return env


def _run_training_once(session: dict[str, Any], epoch: int) -> int:
    env = _child_env(session, epoch)
    completed = subprocess.run(
        [sys.executable, "-m", "zust_runner"],
        env=env,
        check=False,
    )
    return int(completed.returncode or 0)


def _post_idle_workspace_snapshot(client: httpx.Client) -> None:
    from zust_runner.main import resolve_workspace_root
    from zust_runner.outbound import RunOutbound
    from zust_runner.workspace_snapshot import workspace_snapshot_payload

    cache_dir = resolve_workspace_root() / "outbound"
    outbound = RunOutbound(
        client,
        api_base=_api_base(),
        run_id=_run_id(),
        headers=_headers(),
        cache_dir=cache_dir,
    )
    try:
        outbound.post_workspace_snapshot(workspace_snapshot_payload())
    except Exception:
        pass


def _wait_for_work(client: httpx.Client, *, after_epoch: int) -> dict[str, Any]:
    last_snapshot = 0.0
    while True:
        session = fetch_session(client)
        if session.get("cancel"):
            return session
        epoch = int(session.get("runner_epoch") or 1)
        if epoch > after_epoch:
            return session
        if session.get("pending_action"):
            return session
        status = str(session.get("execution_status") or "")
        if status == "CANCELLED":
            return session
        now = time.monotonic()
        if now - last_snapshot >= 30.0:
            _post_idle_workspace_snapshot(client)
            last_snapshot = now
        time.sleep(POLL_SECONDS)


def main() -> int:
    if not (_api_base() and _run_id() and os.environ.get("ZUST_RUN_TOKEN")):
        print("Missing ZUST_API_BASE, ZUST_RUN_ID, or ZUST_RUN_TOKEN", file=sys.stderr)
        return 2

    active_epoch = 0
    adopted_epoch = False
    with create_runner_client() as client:
        while True:
            session = fetch_session(client)
            if session.get("cancel") or session.get("execution_status") == "CANCELLED":
                return 0

            target_epoch = int(session.get("runner_epoch") or 1)
            epoch_changed = adopted_epoch and target_epoch != active_epoch
            adopted_epoch = True
            active_epoch = target_epoch

            if _pending_type(session) == "purge_workspace":
                _finish_purge(client)
                if not epoch_changed:
                    _wait_for_work(client, after_epoch=active_epoch)
                    continue
                session = fetch_session(client)

            if _pending_type(session) == "huggingface_push":
                _maybe_run_pending_action(client, session)
                session = fetch_session(client)

            code = _run_training_once(session, active_epoch)
            if code == RESTART_EXIT_CODE:
                continue
            if code == PURGE_EXIT_CODE:
                session = fetch_session(client)
                if _pending_type(session) == "purge_workspace":
                    _finish_purge(client)
                    session = fetch_session(client)
                if session.get("cancel") or session.get("execution_status") == "CANCELLED":
                    return 0
                if int(session.get("runner_epoch") or 1) > active_epoch:
                    continue
                _wait_for_work(client, after_epoch=active_epoch)
                continue
            if code != 0:
                session = fetch_session(client)
                if int(session.get("runner_epoch") or 1) > active_epoch:
                    continue
                if session.get("turn_off_when_finished"):
                    return code
                _wait_for_work(client, after_epoch=active_epoch)
                continue

            session = fetch_session(client)
            if session.get("turn_off_when_finished"):
                return 0
            _wait_for_work(client, after_epoch=active_epoch)


if __name__ == "__main__":
    raise SystemExit(main())
