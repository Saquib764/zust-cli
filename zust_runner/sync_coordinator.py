"""Batch logs, metrics, and heartbeats into periodic POST /sync calls."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from zust_runner.huggingface_push import run_huggingface_push
from zust_runner.metrics import collect_new_scalars, resolve_training_log_dir
from zust_runner.metrics_cursor import apply_ack, load, save
from zust_runner.runner_cache import action_fingerprint

SYNC_INTERVAL_SECONDS = 10.0
WORKSPACE_EVERY_N_SYNCS = 6
METRICS_COLLECT_SECONDS = 3.0
METRICS_FLUSH_TIMEOUT_SECONDS = 120.0
_MAX_LOG_BATCH = 2000
_MAX_METRIC_BATCH = 500


def _metric_key(point: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (point.get("_event_file"), point.get("tag"), point.get("step"))


class RunSyncCoordinator:
    def __init__(
        self,
        outbound: Any,
        *,
        cache_dir: Path,
        checkout: Path | None = None,
        repo_root: Path | None = None,
        config_rel: str = "",
        workspace_payload: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self._outbound = outbound
        self._cache_dir = cache_dir
        self._cursor_path = cache_dir / "metrics-cursor.json"
        self._cursor = load(self._cursor_path)
        self._checkout = checkout
        self._repo_root = repo_root if repo_root is not None else checkout
        self._config_rel = config_rel
        self._workspace_payload = workspace_payload
        self._log_dir: Path | None = None
        if checkout is not None and config_rel:
            self._log_dir = resolve_training_log_dir(
                checkout,
                config_rel,
                repo_root=self._repo_root,
            )

        self._lock = threading.Lock()
        self._log_buffer: list[dict[str, str]] = []
        self._metrics_pending: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._sync_count = 0
        self._thread: threading.Thread | None = None
        self._metrics_thread: threading.Thread | None = None
        self._hf_lock = threading.Lock()
        self._hf_seen: set[tuple[str, str]] = set()
        self._hf_thread: threading.Thread | None = None

    def bind_project_dir(self, project_root: Path, *, repo_root: Path | None = None) -> None:
        """Point metric collection at the training root after the config is written."""
        with self._lock:
            self._checkout = project_root
            if repo_root is not None:
                self._repo_root = repo_root
            self._log_dir = (
                resolve_training_log_dir(
                    project_root,
                    self._config_rel,
                    repo_root=self._repo_root,
                )
                if self._config_rel
                else None
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
        self._join_huggingface_push()
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
        try:
            with self._lock:
                return self._sync_locked(
                    progress=progress,
                    heartbeat=heartbeat,
                    include_workspace=include_workspace,
                )
        finally:
            # A cursor-save failure happens after the API response is stored. The upload
            # still has to start, or a queued Hugging Face push sits forever during training.
            self._maybe_start_huggingface_push()

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
            self._enqueue_metrics(points)

    def _enqueue_metrics(self, points: list[dict[str, Any]]) -> None:
        """Queue each scalar once. A later read refreshes the queued value instead of stacking a copy."""
        index = {_metric_key(point): i for i, point in enumerate(self._metrics_pending)}
        for point in points:
            if self._point_covered(point):
                continue
            key = _metric_key(point)
            existing = index.get(key)
            if existing is not None:
                self._metrics_pending[existing] = point
                continue
            index[key] = len(self._metrics_pending)
            self._metrics_pending.append(point)

    def _point_covered(self, point: dict[str, Any]) -> bool:
        file_key = point.get("_event_file")
        tag = point.get("tag")
        if not isinstance(file_key, str) or not isinstance(tag, str):
            return False
        try:
            step = int(point.get("step"))
        except (TypeError, ValueError):
            return False
        return step <= self._cursor.get(file_key, {}).get(tag, -1)

    def _drop_pending_covered_by_cursor(self) -> None:
        self._metrics_pending = [point for point in self._metrics_pending if not self._point_covered(point)]

    def _sync_loop(self) -> None:
        while not self._stop.wait(SYNC_INTERVAL_SECONDS):
            include_workspace = self._sync_count % WORKSPACE_EVERY_N_SYNCS == 0
            try:
                self.sync_now(heartbeat=True, include_workspace=include_workspace)
            except Exception as exc:
                print(f"Runner sync failed ({exc}); will retry", file=sys.stderr)
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

        try:
            control, metrics_accepted = self._outbound.post_sync(payload)
        except Exception:
            if logs:
                self._log_buffer = logs + self._log_buffer
            if metrics:
                self._metrics_pending = metrics + self._metrics_pending
            raise
        if metrics_accepted == len(metrics) and metrics:
            apply_ack(self._cursor, metrics)
            self._drop_pending_covered_by_cursor()
            try:
                save(self._cursor_path, self._cursor)
            except OSError as exc:
                print(f"Could not save metrics cursor ({exc}); will retry", file=sys.stderr)
        elif metrics:
            self._metrics_pending = metrics + self._metrics_pending
        return control

    def _join_huggingface_push(self) -> None:
        thread = self._hf_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5)

    def _maybe_start_huggingface_push(self) -> None:
        """Upload in the background so a running training process is left alone."""
        if self._stop.is_set():
            return
        peek = getattr(self._outbound, "peek_huggingface_action", None)
        if peek is None:
            return
        action = peek()
        if not isinstance(action, dict) or action.get("type") != "huggingface_push":
            return
        fingerprint = (str(action.get("type") or ""), str(action.get("requested_at") or ""))
        with self._hf_lock:
            if fingerprint in self._hf_seen:
                return
            if self._hf_thread is not None and self._hf_thread.is_alive():
                return
            self._hf_seen.add(fingerprint)
            self._hf_thread = threading.Thread(
                target=self._huggingface_worker,
                args=(dict(action),),
                name="huggingface-push",
                daemon=True,
            )
            self._hf_thread.start()

    def _huggingface_worker(self, action: dict[str, Any]) -> None:
        checkout = self._repo_root
        try:
            if checkout is None:
                raise RuntimeError("training repo path is not ready")
            run_huggingface_push(action, checkout, log=lambda line: self.buffer_log("stdout", line))
        except Exception as exc:
            self.buffer_log("stderr", f"Hugging Face upload failed: {exc}")
        try:
            self.sync_now(heartbeat=False)
        except Exception:
            pass
        current = self._outbound.peek_huggingface_action()
        if not isinstance(current, dict) or action_fingerprint(current) != action_fingerprint(action):
            return
        complete = getattr(self._outbound, "complete_runner_action", None)
        if complete is None:
            return
        try:
            complete(action)
        except Exception as exc:
            self.buffer_log("stderr", f"Hugging Face upload finished, but the queue was not cleared: {exc}")
            self.sync_now(heartbeat=False)
