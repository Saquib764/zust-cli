"""Runner path checks."""

import tempfile
from pathlib import Path

import pytest

from zust_runner.main import (
    _validate_repo_relative_path,
    _write_repo_dotenv,
    resolve_workspace_root,
)


def test_workspace_root_uses_callback_run_directory(monkeypatch):
    monkeypatch.setenv("ZUST_RUN_ID", "run_abc")
    monkeypatch.setenv("ZUST_WORKSPACE_ROOT", "/tmp/zust-workspace/run_abc")
    assert resolve_workspace_root() == Path("/tmp/zust-workspace/run_abc")


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


def test_runner_rejects_traversal_path():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with pytest.raises(ValueError):
            _validate_repo_relative_path(root, "../outside.yaml")
