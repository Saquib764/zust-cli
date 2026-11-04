"""Tail TensorBoard event files and POST scalar points to the Zust API."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import httpx
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

_MAX_TAG_LEN = 200
_MAX_POINTS = 500
_POLL_SECONDS = 3.0


def resolve_training_log_dir(checkout: Path, config_rel: str) -> Path | None:
    normalized = config_rel.replace("\\", "/").lstrip("/")
    if ".." in normalized.split("/"):
        return None
    config_file = (checkout / normalized).resolve()
    checkout_resolved = checkout.resolve()
    if config_file != checkout_resolved and checkout_resolved not in config_file.parents:
        return None
    if not config_file.is_file():
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
    if log_dir != checkout_resolved and checkout_resolved not in log_dir.parents:
        return None
    return log_dir


def _collect_points(log_dir: Path, sent_max_step: dict[str, int]) -> list[dict[str, Any]]:
    if not log_dir.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for event_file in sorted(log_dir.glob("events.out.tfevents.*")):
        if not event_file.is_file():
            continue
        try:
            accumulator = EventAccumulator(str(event_file))
            accumulator.Reload()
        except Exception:
            continue
        for tag in accumulator.Tags().get("scalars", []):
            if len(tag) > _MAX_TAG_LEN:
                continue
            last = sent_max_step.get(tag, -1)
            max_step = last
            for event in accumulator.Scalars(tag):
                step = int(event.step)
                if step <= last:
                    continue
                value = float(event.value)
                if value != value or value in (float("inf"), float("-inf")):
                    continue
                point: dict[str, Any] = {"tag": tag, "step": step, "value": value}
                if event.wall_time:
                    point["wall_time"] = float(event.wall_time)
                out.append(point)
                max_step = max(max_step, step)
            if max_step > last:
                sent_max_step[tag] = max_step
    out.sort(key=lambda item: (item["tag"], item["step"]))
    return out


def _append_metrics(
    client: httpx.Client,
    run_id: str,
    headers: dict[str, str],
    api_base: str,
    points: list[dict[str, Any]],
    send_with_retries,
) -> None:
    if not points:
        return
    for start in range(0, len(points), _MAX_POINTS):
        batch = points[start : start + _MAX_POINTS]

        def _send(batch=batch):
            client.post(
                f"{api_base}/api/v1/internal/runs/{run_id}/metrics",
                headers=headers,
                json={"points": batch},
                timeout=30.0,
            ).raise_for_status()

        send_with_retries(_send)


def tail_training_metrics(
    client: httpx.Client,
    *,
    api_base: str,
    run_id: str,
    headers: dict[str, str],
    log_dir: Path,
    stop: threading.Event,
    send_with_retries,
) -> None:
    sent_max_step: dict[str, int] = {}
    while not stop.wait(_POLL_SECONDS):
        points = _collect_points(log_dir, sent_max_step)
        _append_metrics(client, run_id, headers, api_base, points, send_with_retries)
    points = _collect_points(log_dir, sent_max_step)
    _append_metrics(client, run_id, headers, api_base, points, send_with_retries)


def run_training_with_metrics(
    client: httpx.Client,
    argv: list[str],
    checkout: Path,
    config_rel: str,
    *,
    api_base: str,
    run_id: str,
    headers: dict[str, str],
    run_argv,
    send_with_retries,
) -> int:
    log_dir = resolve_training_log_dir(checkout, config_rel)
    stop = threading.Event()
    tail_thread: threading.Thread | None = None
    if log_dir is not None:
        tail_thread = threading.Thread(
            target=tail_training_metrics,
            args=(client,),
            kwargs={
                "api_base": api_base,
                "run_id": run_id,
                "headers": headers,
                "log_dir": log_dir,
                "stop": stop,
                "send_with_retries": send_with_retries,
            },
            daemon=True,
        )
        tail_thread.start()
    try:
        return run_argv(client, argv, checkout)
    finally:
        if tail_thread is not None:
            stop.set()
            tail_thread.join(timeout=60)
