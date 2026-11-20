"""Runner path checks."""

import sys
import tempfile
from pathlib import Path

import pytest

from zust_runner.main import (
    StreamSplitter,
    _clone_commit,
    _github_remote_url,
    _training_workdir,
    scrub_github_credentials,
    _image_torch_steps,
    _validate_repo_relative_path,
    _with_uv_skip_image_torch_packages,
    _write_repo_dotenv,
    effective_setup_argv,
    normalize_uv_argv,
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


def test_training_workdir_defaults_to_checkout(monkeypatch, tmp_path):
    monkeypatch.delenv("ZUST_ROOT_PATH", raising=False)
    checkout = tmp_path / "repo"
    checkout.mkdir()
    assert _training_workdir(checkout) == checkout


def test_training_workdir_uses_repo_subfolder(monkeypatch, tmp_path):
    monkeypatch.setenv("ZUST_ROOT_PATH", "train")
    checkout = tmp_path / "repo"
    workdir = checkout / "train"
    workdir.mkdir(parents=True)
    assert _training_workdir(checkout) == workdir


def test_training_workdir_rejects_missing_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("ZUST_ROOT_PATH", "missing")
    checkout = tmp_path / "repo"
    checkout.mkdir()
    with pytest.raises(ValueError, match="Root folder not found"):
        _training_workdir(checkout)


def test_github_remote_url_is_anonymous_without_token(monkeypatch):
    monkeypatch.delenv("ZUST_GITHUB_TOKEN", raising=False)
    assert _github_remote_url("acme", "models") == "https://github.com/acme/models.git"


def test_scrub_github_credentials_redacts_remote_token():
    line = "fatal: https://x-access-token:gh_secret@github.com/acme/models.git not found"
    assert "gh_secret" not in scrub_github_credentials(line)
    assert "x-access-token:redacted@" in scrub_github_credentials(line)


def test_clone_commit_removes_token_from_origin(monkeypatch, tmp_path):
    calls: list[list[str]] = []
    monkeypatch.setattr("zust_runner.main.REPOSITORY", "acme/models")
    monkeypatch.setattr("zust_runner.main.GIT_BRANCH", "main")
    monkeypatch.setattr("zust_runner.main.COMMIT_SHA", "abc")
    monkeypatch.setenv("ZUST_GITHUB_TOKEN", "gh_secret")
    monkeypatch.delenv("ZUST_WARM_RESTART", raising=False)

    def record(_outbound, argv, cwd):
        calls.append(argv)
        if argv[:2] == ["git", "init"]:
            (cwd / ".git").mkdir()
        return 0

    monkeypatch.setattr("zust_runner.main._run_argv", record)
    _clone_commit(tmp_path, outbound=None)
    assert calls[-1] == ["git", "remote", "set-url", "origin", "https://github.com/acme/models.git"]
    assert any("x-access-token:gh_secret@" in part for argv in calls for part in argv)


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


def test_image_torch_sync_skips_torch_install(monkeypatch, tmp_path):
    monkeypatch.setenv("ZUST_USE_IMAGE_TORCH", "1")
    prelude, argv = _image_torch_steps(["uv", "sync"], tmp_path)
    assert prelude == [
        ["uv", "venv", "--python", sys.executable, "--system-site-packages", ".venv"]
    ]
    assert argv == _with_uv_skip_image_torch_packages(["uv", "sync"])


def test_image_torch_run_reuses_venv_without_syncing_torch(monkeypatch, tmp_path):
    monkeypatch.setenv("ZUST_USE_IMAGE_TORCH", "1")
    venv = tmp_path / ".venv"
    venv.mkdir()
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\ninclude-system-site-packages = true\n")
    prelude, argv = _image_torch_steps(["uv", "run", "gemma-author-probe", "train"], tmp_path)
    assert prelude == [_with_uv_skip_image_torch_packages(["uv", "sync"])]
    assert argv == ["uv", "run", "--no-sync", "gemma-author-probe", "train"]


def test_image_torch_off_leaves_uv_commands_unchanged(monkeypatch, tmp_path):
    monkeypatch.delenv("ZUST_USE_IMAGE_TORCH", raising=False)
    prelude, argv = _image_torch_steps(["uv", "sync"], tmp_path)
    assert prelude == []
    assert argv == ["uv", "sync"]


def test_normalize_uv_argv_wraps_python_when_pyproject_exists(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = \"demo\"\n")
    argv = ["python", "-u", "train.py", "--config", "config.yaml"]
    assert normalize_uv_argv(argv, checkout=tmp_path) == [
        "uv",
        "run",
        "python",
        "-u",
        "train.py",
        "--config",
        "config.yaml",
    ]


def test_normalize_uv_argv_leaves_uv_commands_unchanged(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = \"demo\"\n")
    argv = ["uv", "run", "python", "train.py"]
    assert normalize_uv_argv(argv, checkout=tmp_path) == argv


def test_effective_setup_argv_defaults_to_uv_sync(tmp_path, monkeypatch):
    monkeypatch.setenv("ZUST_SETUP_ARGV", "[]")
    monkeypatch.delenv("ZUST_SKIP_SETUP", raising=False)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = \"demo\"\n")
    assert effective_setup_argv(tmp_path) == ["uv", "sync"]


def test_runner_rejects_traversal_path():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with pytest.raises(ValueError):
            _validate_repo_relative_path(root, "../outside.yaml")
