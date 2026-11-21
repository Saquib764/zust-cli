"""Read TensorBoard event files for runner sync batches."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from zust_runner.config_paths import _path_within_any_anchor, find_config_file

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


def _prune_event_file_state(cursor: dict[str, dict[str, int]], active_files: set[str]) -> None:
    for stale in [name for name in cursor if name not in active_files]:
        del cursor[stale]


def collect_new_scalars(log_dir: Path, cursor: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    """Return scalar points newer than the persisted cursor (cursor is not mutated)."""
    if not log_dir.is_dir():
        return []
    event_files = sorted(path for path in log_dir.glob("events.out.tfevents.*") if path.is_file())
    active_names = {path.name for path in event_files}
    _prune_event_file_state(cursor, active_names)

    out: list[dict[str, Any]] = []
    for event_file in event_files:
        file_key = event_file.name
        tag_state = cursor.get(file_key, {})
        try:
            accumulator = EventAccumulator(str(event_file))
            accumulator.Reload()
        except Exception:
            continue
        for tag in accumulator.Tags().get("scalars", []):
            if len(tag) > _MAX_TAG_LEN:
                continue
            last = tag_state.get(tag, -1)
            for event in accumulator.Scalars(tag):
                step = int(event.step)
                if step <= last:
                    continue
                value = float(event.value)
                if value != value or value in (float("inf"), float("-inf")):
                    continue
                point: dict[str, Any] = {
                    "_event_file": file_key,
                    "tag": tag,
                    "step": step,
                    "value": value,
                }
                if event.wall_time:
                    point["wall_time"] = float(event.wall_time)
                out.append(point)
    out.sort(key=lambda item: (item["tag"], item["step"]))
    return out


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
