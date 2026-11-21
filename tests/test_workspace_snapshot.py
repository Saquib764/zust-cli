from pathlib import Path

from zust_runner.workspace_snapshot import scan_workspace_root


def test_scan_skips_dot_prefixed_paths(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run_hidden"
    repo = run_dir / "repo"
    repo.mkdir(parents=True)
    (repo / ".env").write_text("x", encoding="utf-8")
    (repo / ".cache").mkdir()
    (repo / ".cache" / "token").write_text("t", encoding="utf-8")
    (repo / "train.py").write_text("ok", encoding="utf-8")
    (repo / "pkg").mkdir()
    (repo / "pkg" / ".secret").write_text("n", encoding="utf-8")
    (repo / "pkg" / "main.py").write_text("ok", encoding="utf-8")

    monkeypatch.setenv("ZUST_WORKSPACE_ROOT", str(run_dir))

    paths = {row["relative_path"] for row in scan_workspace_root()}
    assert ".env" not in paths
    assert ".cache" not in paths
    assert "pkg/.secret" not in paths
    assert "train.py" in paths
    assert "pkg" in paths
    assert "pkg/main.py" in paths