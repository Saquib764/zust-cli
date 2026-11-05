"""Persist runner session and config on the pod when the API is temporarily down."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx

from zust_runner.http_client import is_retriable_api_error, merge_runner_headers, with_transport_retries


def resolve_workspace_root() -> Path:
    override = os.environ.get("ZUST_WORKSPACE_ROOT", "").strip()
    if override:
        return Path(override)
    return Path("/workspace/runs") / os.environ.get("ZUST_RUN_ID", "")

SessionSource = Literal["live", "cache", "unavailable"]

SESSION_FILE = "session.json"
ACTION_COMPLETE_PENDING_FILE = "action-complete-pending.json"
EFFECTIVE_CONFIG_FILE = "effective-config"
EFFECTIVE_ENV_FILE = "effective-env"

API_POLL_SECONDS = 3.0


@dataclass(frozen=True)
class SessionPoll:
    body: dict[str, Any] | None
    source: SessionSource

    @property
    def live(self) -> bool:
        return self.source == "live"


def outbound_dir() -> Path:
    path = resolve_workspace_root() / "outbound"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_bytes_atomic(destination: Path, content: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=destination.parent) as handle:
        handle.write(content)
        temp_name = handle.name
    os.replace(temp_name, destination)


def write_json_atomic(destination: Path, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    _write_bytes_atomic(destination, data)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, dict) else None


def action_fingerprint(action: dict[str, Any]) -> dict[str, str]:
    requested_at = action.get("requested_at")
    action_type = action.get("type")
    if isinstance(action_type, str) and isinstance(requested_at, str):
        return {"type": action_type, "requested_at": requested_at}
    return {"type": str(action_type or ""), "requested_at": str(requested_at or "")}


def pending_action_matches(session_action: Any, fingerprint: dict[str, str]) -> bool:
    if not isinstance(session_action, dict):
        return False
    return action_fingerprint(session_action) == fingerprint


def save_action_complete_pending(action: dict[str, Any]) -> None:
    write_json_atomic(outbound_dir() / ACTION_COMPLETE_PENDING_FILE, action_fingerprint(action))


def clear_action_complete_pending() -> None:
    path = outbound_dir() / ACTION_COMPLETE_PENDING_FILE
    if path.exists():
        path.unlink()


def load_action_complete_pending() -> dict[str, str] | None:
    return read_json(outbound_dir() / ACTION_COMPLETE_PENDING_FILE)  # type: ignore[return-value]


def save_session(body: dict[str, Any]) -> None:
    write_json_atomic(outbound_dir() / SESSION_FILE, body)


def load_session() -> dict[str, Any] | None:
    return read_json(outbound_dir() / SESSION_FILE)


def fetch_session_once(
    client: httpx.Client,
    *,
    api_base: str,
    run_id: str,
    headers: dict[str, str],
) -> SessionPoll:
    url = f"{api_base.rstrip('/')}/api/v1/internal/runs/{run_id}/session"
    try:
        response = client.get(url, headers=merge_runner_headers(headers), timeout=60.0)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise httpx.HTTPError("session response was not an object")
        save_session(body)
        return SessionPoll(body=body, source="live")
    except httpx.HTTPStatusError as exc:
        if not is_retriable_api_error(exc):
            raise
    except (httpx.TransportError, httpx.TimeoutException):
        pass

    cached = load_session()
    if cached is not None:
        return SessionPoll(body=cached, source="cache")
    return SessionPoll(body=None, source="unavailable")


def post_action_complete(
    client: httpx.Client,
    *,
    api_base: str,
    run_id: str,
    headers: dict[str, str],
) -> None:
    url = f"{api_base.rstrip('/')}/api/v1/internal/runs/{run_id}/action-complete"

    def _post() -> None:
        client.post(url, headers=merge_runner_headers(headers), timeout=60.0).raise_for_status()

    with_transport_retries(_post)
    clear_action_complete_pending()


def complete_action_resilient(
    client: httpx.Client,
    *,
    api_base: str,
    run_id: str,
    headers: dict[str, str],
    ack_action: dict[str, Any] | None = None,
) -> None:
    try:
        post_action_complete(client, api_base=api_base, run_id=run_id, headers=headers)
    except Exception as exc:
        if is_retriable_api_error(exc) and ack_action is not None:
            save_action_complete_pending(ack_action)
            return
        raise


def pending_ack_matches_action(action: dict[str, Any]) -> bool:
    pending = load_action_complete_pending()
    if pending is None:
        return False
    return action_fingerprint(action) == pending


def replay_action_complete_if_needed(
    client: httpx.Client,
    *,
    api_base: str,
    run_id: str,
    headers: dict[str, str],
    live_session: dict[str, Any],
) -> None:
    pending = load_action_complete_pending()
    if pending is None:
        return
    if not pending_action_matches(live_session.get("pending_action"), pending):
        clear_action_complete_pending()
        return
    try:
        post_action_complete(client, api_base=api_base, run_id=run_id, headers=headers)
    except Exception as exc:
        if is_retriable_api_error(exc):
            return
        raise


def fetch_cached_bytes(
    client: httpx.Client,
    *,
    url: str,
    headers: dict[str, str],
    cache_path: Path,
    poll_seconds: float = API_POLL_SECONDS,
) -> bytes:
    while True:
        try:
            response = client.get(url, headers=merge_runner_headers(headers), timeout=60.0)
            response.raise_for_status()
            _write_bytes_atomic(cache_path, response.content)
            return response.content
        except httpx.HTTPStatusError as exc:
            if not is_retriable_api_error(exc):
                raise
        except (httpx.TransportError, httpx.TimeoutException):
            pass
        if cache_path.exists():
            return cache_path.read_bytes()
        time.sleep(poll_seconds)
