"""Remove TensorBoard event files before a warm restart."""

from __future__ import annotations

from pathlib import Path

from zust_runner.metrics import resolve_training_log_dir


def clean_tensorboard_events(checkout: Path, config_rel: str) -> int:
    """Delete events.out.tfevents.* under the training log_dir. Returns files removed."""
    log_dir = resolve_training_log_dir(checkout, config_rel)
    if log_dir is None or not log_dir.is_dir():
        return 0
    removed = 0
    for event_file in log_dir.glob("events.out.tfevents.*"):
        if not event_file.is_file():
            continue
        try:
            event_file.unlink()
            removed += 1
        except OSError:
            continue
    return removed
