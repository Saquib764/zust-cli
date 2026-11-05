"""Session and artifact cache when the API is temporarily unavailable."""

import json
from pathlib import Path

import httpx
import pytest

from zust_runner.runner_cache import (
    ACTION_COMPLETE_PENDING_FILE,
    complete_action_resilient,
    fetch_cached_bytes,
    fetch_session_once,
    load_session,
    outbound_dir,
    replay_action_complete_if_needed,
    save_session,
)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("ZUST_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("ZUST_RUN_ID", "run_test")
    return tmp_path


def test_fetch_session_returns_cache_on_502(workspace: Path) -> None:
    save_session({"runner_epoch": 2, "cancel": False, "execution_status": "RUNNING"})

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(502)

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.test")
    poll = fetch_session_once(
        client,
        api_base="https://api.test",
        run_id="run_test",
        headers={"Authorization": "Bearer t"},
    )
    assert poll.source == "cache"
    assert poll.body is not None
    assert poll.body["runner_epoch"] == 2


def test_fetch_session_saves_live_response(workspace: Path) -> None:
    payload = {"runner_epoch": 3, "cancel": False, "execution_status": "RUNNING"}

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.test")
    poll = fetch_session_once(
        client,
        api_base="https://api.test",
        run_id="run_test",
        headers={"Authorization": "Bearer t"},
    )
    assert poll.source == "live"
    assert poll.body == payload
    cached = load_session()
    assert cached == payload


def test_fetch_session_unavailable_without_cache(workspace: Path) -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _request: httpx.Response(502)),
        base_url="https://api.test",
    )
    poll = fetch_session_once(
        client,
        api_base="https://api.test",
        run_id="run_test",
        headers={"Authorization": "Bearer t"},
    )
    assert poll.source == "unavailable"
    assert poll.body is None


def test_fetch_session_raises_on_403(workspace: Path) -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _request: httpx.Response(403)),
        base_url="https://api.test",
    )
    with pytest.raises(httpx.HTTPStatusError):
        fetch_session_once(
            client,
            api_base="https://api.test",
            run_id="run_test",
            headers={"Authorization": "Bearer t"},
        )


def test_complete_action_persists_pending_on_502(workspace: Path) -> None:
    action = {"type": "purge_workspace", "requested_at": "2026-01-01T00:00:00Z"}
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _request: httpx.Response(502)),
        base_url="https://api.test",
    )
    complete_action_resilient(
        client,
        api_base="https://api.test",
        run_id="run_test",
        headers={"Authorization": "Bearer t"},
        ack_action=action,
    )
    pending_path = outbound_dir() / ACTION_COMPLETE_PENDING_FILE
    assert pending_path.exists()
    assert json.loads(pending_path.read_text()) == {
        "type": "purge_workspace",
        "requested_at": "2026-01-01T00:00:00Z",
    }


def test_replay_action_complete_clears_matching_pending(workspace: Path) -> None:
    pending_path = outbound_dir() / ACTION_COMPLETE_PENDING_FILE
    pending_path.parent.mkdir(parents=True, exist_ok=True)
    pending_path.write_text(
        json.dumps({"type": "purge_workspace", "requested_at": "2026-01-01T00:00:00Z"}),
        encoding="utf-8",
    )
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/action-complete"):
            calls.append("ack")
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.test")
    replay_action_complete_if_needed(
        client,
        api_base="https://api.test",
        run_id="run_test",
        headers={"Authorization": "Bearer t"},
        live_session={
            "pending_action": {"type": "purge_workspace", "requested_at": "2026-01-01T00:00:00Z"},
        },
    )
    assert calls == ["ack"]
    assert not pending_path.exists()


def test_fetch_cached_bytes_uses_disk_on_502(workspace: Path) -> None:
    cache_path = outbound_dir() / "effective-config"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(b"train:\n  epochs: 1\n")
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _request: httpx.Response(502)),
        base_url="https://api.test",
    )
    content = fetch_cached_bytes(
        client,
        url="https://api.test/api/v1/internal/runs/run_test/effective-config",
        headers={"Authorization": "Bearer t"},
        cache_path=cache_path,
        poll_seconds=0.01,
    )
    assert content == b"train:\n  epochs: 1\n"
