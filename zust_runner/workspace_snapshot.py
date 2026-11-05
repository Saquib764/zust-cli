"""Scan the training repo and POST a flat file index to the API."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from zust_runner.main import resolve_workspace_root

_MAX_ENTRIES = 8000
_SKIP_DIR_NAMES = frozenset({"__pycache__", ".git", ".venv"})


def repo_root() -> Path:
    return resolve_workspace_root() / "repo"


def scan_workspace_root() -> list[dict[str, Any]]:
    root = repo_root()
    if not root.is_dir():
        return []
    entries: list[dict[str, Any]] = []
    seen_dirs: set[str] = set()

    def add_directory(relative: str, mtime: float | None = None) -> None:
        if relative in seen_dirs:
            return
        seen_dirs.add(relative)
        row: dict[str, Any] = {"relative_path": relative, "kind": "directory"}
        if mtime is not None:
            row["last_modified"] = datetime.fromtimestamp(mtime, tz=UTC).isoformat()
        entries.append(row)

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in _SKIP_DIR_NAMES]
        current = Path(dirpath)
        try:
            rel_dir = current.relative_to(root).as_posix()
        except ValueError:
            continue
        if rel_dir and rel_dir != ".":
            add_directory(rel_dir, current.stat().st_mtime)

        for name in filenames:
            if name.startswith("."):
                continue
            file_path = current / name
            try:
                rel_file = file_path.relative_to(root).as_posix()
            except ValueError:
                continue
            try:
                stat = file_path.stat()
            except OSError:
                continue
            entries.append(
                {
                    "relative_path": rel_file,
                    "kind": "file",
                    "size_bytes": stat.st_size,
                    "last_modified": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
                }
            )
            if len(entries) >= _MAX_ENTRIES:
                return entries

    entries.sort(key=lambda row: (0 if row["kind"] == "directory" else 1, row["relative_path"].lower()))
    return entries


def workspace_snapshot_payload() -> dict[str, Any]:
    root = repo_root()
    run_root = resolve_workspace_root()
    entries = scan_workspace_root()
    return {
        "root": str(root) if root.is_dir() else str(run_root / "repo"),
        "entries": entries,
        "truncated": len(entries) >= _MAX_ENTRIES,
    }
