"""Hugging Face uploads during an active training sync loop."""

from pathlib import Path

import pytest

from zust_runner.huggingface_push import _token
from zust_runner.sync_coordinator import RunSyncCoordinator


def test_token_reads_effective_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZUST_WORKSPACE_ROOT", str(tmp_path))
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    env_file = tmp_path / "outbound" / "effective-env"
    env_file.parent.mkdir(parents=True)
    env_file.write_text('export HF_TOKEN="hf_from_run_env"\n', encoding="utf-8")

    assert _token(tmp_path / "repo") == "hf_from_run_env"


def test_token_reads_repo_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZUST_WORKSPACE_ROOT", str(tmp_path))
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    checkout = tmp_path / "repo"
    checkout.mkdir()
    (checkout / ".env").write_text("HF_TOKEN=hf_from_dotenv\n", encoding="utf-8")

    assert _token(checkout) == "hf_from_dotenv"


def test_token_missing_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZUST_WORKSPACE_ROOT", str(tmp_path))
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        _token(tmp_path / "repo")


class _Outbound:
    def __init__(self, action: dict | None) -> None:
        self.action = action
        self.payloads: list[dict] = []
        self.completed: list[dict] = []

    def post_sync(self, payload: dict) -> tuple[int, int]:
        self.payloads.append(payload)
        return 0, len(payload.get("metric_points") or [])

    def peek_huggingface_action(self) -> dict | None:
        return self.action

    def complete_runner_action(self, action: dict) -> None:
        self.completed.append(action)
        self.action = None


def test_sync_starts_huggingface_push_without_stopping_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    action = {
        "type": "huggingface_push",
        "folder_path": "outputs/checkpoint",
        "repo_id": "user/model",
        "requested_at": "2026-11-21T00:00:00+00:00",
    }
    calls: list[dict] = []

    def fake_push(pending: dict, checkout: Path, *, log) -> None:
        calls.append(pending)
        log(f"Uploading {pending['folder_path']}")

    monkeypatch.setattr("zust_runner.sync_coordinator.run_huggingface_push", fake_push)
    outbound = _Outbound(action)
    checkout = tmp_path / "repo"
    checkout.mkdir()
    coordinator = RunSyncCoordinator(outbound, cache_dir=tmp_path / "cache", checkout=checkout)

    assert coordinator.sync_now(heartbeat=True) == 0
    assert coordinator._hf_thread is not None
    coordinator._hf_thread.join(timeout=5)

    assert calls == [action]
    assert outbound.completed == [action]
    assert any("Uploading outputs/checkpoint" in entry["line"] for payload in outbound.payloads for entry in payload["log_entries"])

    coordinator.sync_now(heartbeat=True)
    assert calls == [action]


def test_huggingface_push_starts_when_sync_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    action = {
        "type": "huggingface_push",
        "folder_path": "outputs/checkpoint",
        "repo_id": "user/model",
        "requested_at": "2026-11-21T00:00:00+00:00",
    }
    calls: list[dict] = []

    def fake_push(pending: dict, checkout: Path, *, log) -> None:
        del checkout
        calls.append(pending)
        log("uploading")

    def explode(self, **kwargs) -> int:
        del self, kwargs
        raise RuntimeError("cursor save failed")

    monkeypatch.setattr("zust_runner.sync_coordinator.run_huggingface_push", fake_push)
    monkeypatch.setattr(RunSyncCoordinator, "_sync_locked", explode)
    outbound = _Outbound(action)
    checkout = tmp_path / "repo"
    checkout.mkdir()
    coordinator = RunSyncCoordinator(outbound, cache_dir=tmp_path / "cache", checkout=checkout)

    with pytest.raises(RuntimeError, match="cursor save failed"):
        coordinator.sync_now(heartbeat=True)

    assert coordinator._hf_thread is not None
    coordinator._hf_thread.join(timeout=5)
    assert calls == [action]
    assert outbound.completed == [action]


def test_huggingface_push_failure_is_logged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    action = {
        "type": "huggingface_push",
        "folder_path": "missing",
        "repo_id": "user/model",
        "requested_at": "2026-11-21T00:00:00+00:00",
    }

    def fake_push(_pending: dict, _checkout: Path, *, log) -> None:
        raise FileNotFoundError("Folder not found: missing")

    monkeypatch.setattr("zust_runner.sync_coordinator.run_huggingface_push", fake_push)
    outbound = _Outbound(action)
    checkout = tmp_path / "repo"
    checkout.mkdir()
    coordinator = RunSyncCoordinator(outbound, cache_dir=tmp_path / "cache", checkout=checkout)

    coordinator.sync_now(heartbeat=True)
    assert coordinator._hf_thread is not None
    coordinator._hf_thread.join(timeout=5)

    assert outbound.completed == [action]
    assert any(
        entry["stream"] == "stderr" and "Folder not found: missing" in entry["line"]
        for payload in outbound.payloads
        for entry in payload["log_entries"]
    )
