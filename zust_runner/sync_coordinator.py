"""Batch logs, metrics, and heartbeats into periodic POST /sync calls."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable

from zust_runner.metrics import collect_new_scalars, resolve_training_log_dir
from zust_runner.metrics_cursor import apply_ack, load, save

SYNC_INTERVAL_SECONDS = 10.0
WORKSPACE_EVERY_N_SYNCS = 6
METRICS_COLLECT_SECONDS = 3.0
METRICS_FLUSH_TIMEOUT_SECONDS = 120.0
_MAX_LOG_BATCH = 2000
_MAX_METRIC_BATCH = 500


class RunSyncCoordinator:
    def __init__(
        self,
        outbound: Any,
        *,
        cache_dir: Path,
        checkout: Path | None = None,
        config_rel: str = "",
        workspace_payload: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self._outbound = outbound
        self._cache_dir = cache_dir
        self._cursor_path = cache_dir / "metrics-cursor.json"
        self._cursor = load(self._cursor_path)
        self._checkout = checkout
        self._config_rel = config_rel
        self._workspace_payload = workspace_payload
        self._log_dir: Path | None = None
        if checkout is not None and config_rel:
            self._log_dir = resolve_training_log_dir(checkout, config_rel)

        self._lock = threading.Lock()
        self._log_buffer: list[dict[str, str]] = []
        self._metrics_pending: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._sync_count = 0
        self._thread: threading.Thread | None = None
        self._metrics_thread: threading.Thread | None = None

    def bind_project_dir(self, checkout: Path) -> None:
        """Point metric collection at the training root after the config is written."""
        with self._lock:
            self._checkout = checkout
            self._log_dir = (
                resolve_training_log_dir(checkout, self._config_rel) if self._config_rel else None
            )
        self._ensure_metrics_thread()

    def _ensure_metrics_thread(self) -> None:
        if self._log_dir is None or self._metrics_thread is not None or self._stop.is_set():
            return
        self._metrics_thread = threading.Thread(target=self._metrics_collect_loop, daemon=True)
        self._metrics_thread.start()

    def start(self) -> None:
        self._ensure_metrics_thread()
        self._thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._metrics_thread is not None:
            self._metrics_thread.join(timeout=5)
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.flush_metrics_barrier()
        self.sync_now(heartbeat=True, include_workspace=True)

    def buffer_log(self, stream: str, line: str) -> None:
        with self._lock:
            self._log_buffer.append({"stream": stream, "line": line})

    def sync_now(
        self,
        *,
        progress: dict[str, Any] | None = None,
        heartbeat: bool = False,
        include_workspace: bool = False,
    ) -> int:
        with self._lock:
            return self._sync_locked(progress=progress, heartbeat=heartbeat, include_workspace=include_workspace)

    def flush_metrics_barrier(self) -> bool:
        """Drain TensorBoard scalars to the API before marking training complete."""
        if self._log_dir is None:
            return True
        deadline = time.monotonic() + METRICS_FLUSH_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            self._collect_metrics_into_pending()
            accepted = self.sync_now(heartbeat=False)
            if accepted != 0:
                return False
            with self._lock:
                pending = len(self._metrics_pending)
            fresh = collect_new_scalars(self._log_dir, self._cursor)
            if not pending and not fresh:
                return True
            time.sleep(2.0)
        return False

    def _metrics_collect_loop(self) -> None:
        while not self._stop.wait(METRICS_COLLECT_SECONDS):
            self._collect_metrics_into_pending()

    def _collect_metrics_into_pending(self) -> None:
        if self._log_dir is None:
            return
        points = collect_new_scalars(self._log_dir, self._cursor)
        if not points:
            return
        with self._lock:
            self._metrics_pending.extend(points)

    def _sync_loop(self) -> None:
        while not self._stop.wait(SYNC_INTERVAL_SECONDS):
            include_workspace = self._sync_count % WORKSPACE_EVERY_N_SYNCS == 0
            self.sync_now(heartbeat=True, include_workspace=include_workspace)
            self._sync_count += 1

    def _sync_locked(
        self,
        *,
        progress: dict[str, Any] | None,
        heartbeat: bool,
        include_workspace: bool,
    ) -> int:
        logs = self._log_buffer[:_MAX_LOG_BATCH]
        self._log_buffer = self._log_buffer[len(logs) :]

        metrics: list[dict[str, Any]] = []
        while self._metrics_pending and len(metrics) < _MAX_METRIC_BATCH:
            metrics.append(self._metrics_pending.pop(0))

        payload: dict[str, Any] = {
            "heartbeat": heartbeat,
            "log_entries": logs,
            "metric_points": [
                {k: v for k, v in point.items() if k != "_event_file"} for point in metrics
            ],
        }
        if progress:
            payload["progress"] = progress
        if include_workspace and self._workspace_payload is not None:
            try:
                workspace = self._workspace_payload()
                if workspace.get("entries"):
                    payload["workspace"] = workspace
            except Exception:
                pass

        control, metrics_accepted = self._outbound.post_sync(payload)
        if metrics_accepted == len(metrics) and metrics:
            apply_ack(self._cursor, metrics)
            save(self._cursor_path, self._cursor)
        elif metrics:
            self._metrics_pending = metrics + self._metrics_pending
        return control
