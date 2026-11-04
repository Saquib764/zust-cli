"""Runner path checks."""

import tempfile
from pathlib import Path

import pytest

from zust_runner.main import (
    StreamSplitter,
    _github_remote_url,
    _validate_repo_relative_path,
    _write_repo_dotenv,
    resolve_workspace_root,
)


def test_workspace_root_uses_callback_run_directory(monkeypatch):
    monkeypatch.setenv("ZUST_RUN_ID", "run_abc")
    monkeypatch.setenv("ZUST_WORKSPACE_ROOT", "/tmp/zust-workspace/run_abc")
    assert resolve_workspace_root() == Path("/tmp/zust-workspace/run_abc")


def test_github_remote_url_uses_token_when_set(monkeypatch):
    monkeypatch.setenv("ZUST_GITHUB_TOKEN", "gh_secret")
    url = _github_remote_url("acme", "models")
    assert url == "https://x-access-token:gh_secret@github.com/acme/models.git"


def test_github_remote_url_is_anonymous_without_token(monkeypatch):
    monkeypatch.delenv("ZUST_GITHUB_TOKEN", raising=False)
    assert _github_remote_url("acme", "models") == "https://github.com/acme/models.git"


def test_workspace_root_defaults_to_pod_volume(monkeypatch):
    monkeypatch.delenv("ZUST_WORKSPACE_ROOT", raising=False)
    monkeypatch.setenv("ZUST_RUN_ID", "run_abc")
    assert resolve_workspace_root() == Path("/workspace/runs/run_abc")


def test_write_repo_dotenv_skips_empty():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_repo_dotenv(root, b"   \n")
        assert not (root / ".env").exists()


def test_write_repo_dotenv_writes_file():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_repo_dotenv(root, b"HF_TOKEN=x\n")
        assert (root / ".env").read_text() == "HF_TOKEN=x\n"


def test_progress_carriage_return_updates_one_row():
    splitter = StreamSplitter()
    records = splitter.feed("\rtrain 0 loss=3.8")
    records += splitter.feed("\rtrain 1 loss=3.4")
    records += splitter.feed("\rtrain 1 loss=3.4")
    records += splitter.feed("\nsaved step 1\n")
    records += splitter.feed("\rtrain 1 loss=3.4")
    assert records == [
        "\rtrain 0 loss=3.8",
        "\rtrain 1 loss=3.4",
        "\n",
        "saved step 1",
        "\rtrain 1 loss=3.4",
    ]


def test_progress_split_across_reads_stays_one_row():
    splitter = StreamSplitter()
    records = splitter.feed("\rtrain 0")
    records += splitter.feed(" loss=3.8")
    records += splitter.finish()
    assert records == ["\rtrain 0", "\rtrain 0 loss=3.8"]


def test_ordinary_lines_keep_newlines():
    splitter = StreamSplitter()
    records = splitter.feed("hello\n\nworld\r\n")
    assert records == ["hello", "", "world"]


def test_runner_rejects_traversal_path():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with pytest.raises(ValueError):
            _validate_repo_relative_path(root, "../outside.yaml")
