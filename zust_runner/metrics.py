"""Read TensorBoard event files for runner sync batches."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from zust_runner.config_paths import _path_within_any_anchor, find_config_file
from zust_runner.event_tail import read_records, scalars_from_record

_MAX_TAG_LEN = 200


def resolve_training_log_dir(
    project_root: Path,
    config_rel: str,
    *,
    repo_root: Path | None = None,
) -> Path | None:
    config_file = find_config_file(project_root, config_rel, repo_root=repo_root)
    if config_file is None:
        return None
    raw = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return None
    train = raw.get("train") or {}
    if not isinstance(train, dict):
        return None
    log_dir_raw = train.get("log_dir")
    if log_dir_raw is None or not str(log_dir_raw).strip():
        return None
    log_dir = Path(str(log_dir_raw).strip())
    if not log_dir.is_absolute():
        log_dir = (config_file.parent / log_dir).resolve()
    else:
        log_dir = log_dir.resolve()
    anchors = [project_root.resolve()]
    if repo_root is not None:
        anchors.append(repo_root.resolve())
    if not _path_within_any_anchor(log_dir, anchors):
        return None
    return log_dir


def event_relpath(log_dir: Path, path: Path) -> str | None:
    try:
        return path.resolve().relative_to(log_dir.resolve()).as_posix()
    except ValueError:
        return None


def iter_event_files(log_dir: Path) -> list[Path]:
    if not log_dir.is_dir():
        return []
    files = [
        path
        for path in log_dir.rglob("events.out.tfevents.*")
        if path.is_file() and event_relpath(log_dir, path) is not None
    ]
    return sorted(files, key=lambda path: event_relpath(log_dir, path) or "")


def read_new_scalars(
    log_dir: Path,
    read_state: dict[str, dict[str, int]],
    attempt: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse records past read_state. Returns queue items and files whose inode changed.

    read_state is the in-memory tail, ahead of the acked cursor until a POST succeeds.
    A torn record leaves the offset unchanged. Missing files stay in read_state.
    """
    if not log_dir.is_dir():
        return [], []
    items: list[dict[str, Any]] = []
    reset: list[str] = []
    for event_file in iter_event_files(log_dir):
        rel = event_relpath(log_dir, event_file)
        if rel is None:
            continue
        try:
            stat = event_file.stat()
        except OSError:
            continue
        entry = read_state.setdefault(rel, {"offset": 0, "dev": stat.st_dev, "ino": stat.st_ino})
        if int(entry.get("dev") or 0) != stat.st_dev or int(entry.get("ino") or 0) != stat.st_ino:
            entry["offset"] = 0
            entry["dev"] = stat.st_dev
            entry["ino"] = stat.st_ino
            reset.append(rel)
        offset = int(entry.get("offset") or 0)
        if stat.st_size < offset:
            entry["offset"] = 0
            offset = 0
            reset.append(rel)
        if stat.st_size <= offset:
            continue
        try:
            records, new_offset = read_records(event_file, offset)
        except OSError:
            continue
        for payload, end in records:
            scalars = scalars_from_record(payload)
            scalars = [point for point in scalars if len(str(point["tag"])) <= _MAX_TAG_LEN]
            if not scalars:
                items.append(_marker(rel, end, stat.st_dev, stat.st_ino))
                continue
            for point in scalars:
                items.append(
                    {
                        **point,
                        "attempt": attempt,
                        "_file": rel,
                        "_offset": end,
                        "_dev": stat.st_dev,
                        "_ino": stat.st_ino,
                    }
                )
        entry["offset"] = new_offset
    return items, reset


def _marker(rel: str, end: int, dev: int, ino: int) -> dict[str, Any]:
    return {"_advance": True, "_file": rel, "_offset": end, "_dev": dev, "_ino": ino}


def run_training_with_metrics(
    client,
    argv: list[str],
    checkout: Path,
    config_rel: str,
    *,
    run_argv,
    coordinator=None,
) -> int:
    del client, config_rel
    return run_argv(None, argv, checkout)
