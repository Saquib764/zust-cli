"""Keep the Runpod container alive and restart training without reinstalling zust-cli."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from typing import Any

import httpx

from zust_runner.huggingface_push import run_huggingface_push
from zust_runner.http_client import create_runner_client, merge_runner_headers
from zust_runner.main import PURGE_EXIT_CODE, RESTART_EXIT_CODE, resolve_workspace_root
from zust_runner.runner_cache import (
    API_POLL_SECONDS,
    SessionPoll,
    complete_action_resilient,
    fetch_session_once,
    pending_ack_matches_action,
    replay_action_complete_if_needed,
)

POLL_SECONDS = API_POLL_SECONDS
_last_api_down_log: str | None = None


def _headers() -> dict[str, str]:
    token = os.environ.get("ZUST_RUN_TOKEN", "")
    return merge_runner_headers({"Authorization": f"Bearer {token}"})


def _api_base() -> str:
    return os.environ.get("ZUST_API_BASE", "").rstrip("/")


def _run_id() -> str:
    return os.environ.get("ZUST_RUN_ID", "")


def _client_context() -> dict[str, str]:
    return {"api_base": _api_base(), "run_id": _run_id(), "headers": _headers()}


def _log_api_unreachable(reason: str) -> None:
    global _last_api_down_log
    if _last_api_down_log == reason:
        return
    _last_api_down_log = reason
    print(f"API unreachable ({reason}); using cached session or waiting", file=sys.stderr)


def _clear_api_unreachable_log() -> None:
    global _last_api_down_log
    _last_api_down_log = None


def fetch_session(client: httpx.Client) -> SessionPoll:
    poll = fetch_session_once(client, **_client_context())
    if poll.source == "live":
        _clear_api_unreachable_log()
        replay_action_complete_if_needed(client, **_client_context(), live_session=poll.body or {})
    elif poll.source == "cache":
        _log_api_unreachable("cached session")
    else:
        _log_api_unreachable("no session cache")
    return poll


def complete_action(client: httpx.Client, *, ack_action: dict[str, Any] | None = None) -> None:
    complete_action_resilient(client, **_client_context(), ack_action=ack_action)


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


def _pending_action(session: dict[str, Any]) -> dict[str, Any] | None:
    action = session.get("pending_action")
    return action if isinstance(action, dict) else None


def _finish_purge(client: httpx.Client, action: dict[str, Any]) -> None:
    purge_training_workspace()
    complete_action(client, ack_action=action)


def _run_pending_action(client: httpx.Client, session: dict[str, Any]) -> None:
    action = _pending_action(session)
    if action is None:
        return
    action_type = action.get("type")
    if action_type == "purge_workspace":
        if pending_ack_matches_action(action):
            complete_action(client, ack_action=action)
            return
        _finish_purge(client, action)
        return
    if action_type != "huggingface_push":
        return
    if pending_ack_matches_action(action):
        complete_action(client, ack_action=action)
        return
    checkout = resolve_workspace_root() / "repo"
    run_huggingface_push(action, checkout)
    complete_action(client, ack_action=action)


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


def _sleep_poll() -> None:
    time.sleep(POLL_SECONDS)


def _session_cancelled(session: dict[str, Any]) -> bool:
    return bool(session.get("run_cancelled")) or str(session.get("execution_status") or "") == "CANCELLED"


def _wait_for_work(client: httpx.Client, *, after_epoch: int) -> dict[str, Any]:
    last_snapshot = 0.0
    while True:
        poll = fetch_session(client)
        if poll.source == "unavailable":
            _sleep_poll()
            continue
        session = poll.body
        if session is None:
            _sleep_poll()
            continue
        if not poll.live:
            now = time.monotonic()
            if now - last_snapshot >= 30.0:
                _post_idle_workspace_snapshot(client)
                last_snapshot = now
            _sleep_poll()
            continue
        if _session_cancelled(session):
            return session
        epoch = int(session.get("runner_epoch") or 1)
        if epoch > after_epoch:
            return session
        if session.get("pending_action"):
            return session
        now = time.monotonic()
        if now - last_snapshot >= 30.0:
            _post_idle_workspace_snapshot(client)
            last_snapshot = now
        _sleep_poll()


def _require_session(poll: SessionPoll) -> dict[str, Any] | None:
    if poll.source == "unavailable" or poll.body is None:
        _sleep_poll()
        return None
    return poll.body


def main() -> int:
    if not (_api_base() and _run_id() and os.environ.get("ZUST_RUN_TOKEN")):
        print("Missing ZUST_API_BASE, ZUST_RUN_ID, or ZUST_RUN_TOKEN", file=sys.stderr)
        return 2

    active_epoch = 0
    adopted_epoch = False
    with create_runner_client() as client:
        while True:
            poll = fetch_session(client)
            session = _require_session(poll)
            if session is None:
                continue

            if poll.live and _session_cancelled(session):
                return 0

            target_epoch = int(session.get("runner_epoch") or 1)
            epoch_changed = adopted_epoch and target_epoch != active_epoch
            adopted_epoch = True
            active_epoch = target_epoch

            if poll.live and _pending_type(session) is not None:
                _run_pending_action(client, session)
                if _pending_type(session) == "purge_workspace" and not epoch_changed:
                    _wait_for_work(client, after_epoch=active_epoch)
                    continue
                poll = fetch_session(client)
                session = _require_session(poll)
                if session is None:
                    continue

            code = _run_training_once(session, active_epoch)
            if code == RESTART_EXIT_CODE:
                continue
            if code == PURGE_EXIT_CODE:
                poll = fetch_session(client)
                session = _require_session(poll)
                if session is None:
                    continue
                if poll.live and _pending_type(session) is not None:
                    _run_pending_action(client, session)
                    poll = fetch_session(client)
                    session = _require_session(poll)
                    if session is None:
                        continue
                if poll.live and _session_cancelled(session):
                    return 0
                if poll.live and int(session.get("runner_epoch") or 1) > active_epoch:
                    continue
                _wait_for_work(client, after_epoch=active_epoch)
                continue
            if code != 0:
                poll = fetch_session(client)
                session = _require_session(poll)
                if session is None:
                    _wait_for_work(client, after_epoch=active_epoch)
                    continue
                if poll.live and int(session.get("runner_epoch") or 1) > active_epoch:
                    continue
                if poll.live and session.get("turn_off_when_finished"):
                    return code
                _wait_for_work(client, after_epoch=active_epoch)
                continue

            poll = fetch_session(client)
            if poll.source == "unavailable" or poll.body is None:
                _wait_for_work(client, after_epoch=active_epoch)
                continue
            session = poll.body
            if poll.live and session.get("turn_off_when_finished"):
                return 0
            _wait_for_work(client, after_epoch=active_epoch)


if __name__ == "__main__":
    raise SystemExit(main())
