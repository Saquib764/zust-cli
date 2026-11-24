"""Remove TensorBoard event files before a warm restart."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from zust_runner.metrics import event_relpath, iter_event_files, resolve_training_log_dir


def clean_tensorboard_events(
    project_root: Path,
    config_rel: str,
    *,
    repo_root: Path | None = None,
    acked: dict[str, Any] | None = None,
) -> int:
    """Delete event files under the training log_dir.

    When acked is set, a file is removed only after its bytes have been accepted
    by the API. Passing no cursor deletes every event file.
    """
    log_dir = resolve_training_log_dir(project_root, config_rel, repo_root=repo_root)
    if log_dir is None or not log_dir.is_dir():
        return 0
    files = acked.get("files") if isinstance(acked, dict) else None
    removed = 0
    for event_file in iter_event_files(log_dir):
        rel = event_relpath(log_dir, event_file)
        if files is not None:
            if rel is None or rel not in files:
                continue
            try:
                size = event_file.stat().st_size
            except OSError:
                continue
            if int(files[rel].get("offset") or 0) < size:
                continue
        try:
            event_file.unlink()
        except OSError:
            continue
        removed += 1
    return removed
