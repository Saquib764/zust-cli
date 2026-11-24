"""Batch logs, metrics, and heartbeats into periodic POST /sync calls."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from zust_runner.huggingface_push import run_huggingface_push
from zust_runner.metrics import event_relpath, iter_event_files, read_new_scalars, resolve_training_log_dir
from zust_runner.metrics_cursor import apply_file_ack, load, reset_files, save
from zust_runner.runner_cache import action_fingerprint

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
        repo_root: Path | None = None,
        config_rel: str = "",
        workspace_payload: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self._outbound = outbound
        self._cache_dir = cache_dir
        self._cursor_path = cache_dir / "metrics-cursor.json"
        self._cursor = load(self._cursor_path)
        self._read_state = {
            key: dict(entry) for key, entry in self._cursor.get("files", {}).items()
        }
        self._read_lock = threading.Lock()
        self._pause_reader = threading.Event()
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
        if self._log_dir is None or self._stop.is_set() or self._pause_reader.is_set():
            return
        if self._metrics_thread is not None and self._metrics_thread.is_alive():
            return
        self._metrics_thread = threading.Thread(target=self._metrics_collect_loop, daemon=True)
        self._metrics_thread.start()

    def start(self) -> None:
        self._ensure_metrics_thread()
        self._thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._pause_reader.set()
        if self._metrics_thread is not None:
            self._metrics_thread.join(timeout=5)
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._join_huggingface_push()
        self.flush_metrics_barrier()
        self.sync_now(heartbeat=True, include_workspace=True)

    @property
    def cursor(self) -> dict[str, Any]:
        with self._lock:
            return {"version": self._cursor.get("version"), "files": {k: dict(v) for k, v in self._cursor.get("files", {}).items()}}

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
                control = self._sync_locked(
                    progress=progress,
                    heartbeat=heartbeat,
                    include_workspace=include_workspace,
                )
            try:
                self._drain_metric_batches(1)
            except Exception as exc:
                print(f"Runner metric upload failed ({exc}); will retry", file=sys.stderr)
            return control
        finally:
            # A cursor-save failure happens after the API response is stored. The upload
            # still has to start, or a queued Hugging Face push sits forever during training.
            self._maybe_start_huggingface_push()

    def flush_metrics_barrier(self) -> bool:
        """Drain TensorBoard scalars. A cancel or restart does not abandon the queue."""
        if self._log_dir is None:
            return True
        self._pause_reader.set()
        thread = self._metrics_thread
        if thread is not None and thread is not threading.current_thread() and thread.is_alive():
            thread.join(timeout=15)
        deadline = time.monotonic() + METRICS_FLUSH_TIMEOUT_SECONDS
        caught_up = False
        try:
            while time.monotonic() < deadline:
                self._collect_metrics_into_pending()
                try:
                    self._drain_metric_batches(50)
                except Exception as exc:
                    print(f"Runner metric upload failed ({exc}); will retry", file=sys.stderr)
                    time.sleep(1.0)
                    continue
                with self._lock:
                    pending = len(self._metrics_pending)
                if pending == 0 and not self._log_has_unread():
                    caught_up = True
                    break
                time.sleep(0.2)
            return caught_up
        finally:
            if not self._stop.is_set():
                self._pause_reader.clear()
                self._ensure_metrics_thread()

    def _metrics_collect_loop(self) -> None:
        while not self._stop.is_set():
            if self._pause_reader.is_set():
                return
            try:
                self._collect_metrics_into_pending()
                self._drain_metric_batches(4)
            except Exception as exc:
                print(f"Runner metric upload failed ({exc}); will retry", file=sys.stderr)
            if self._pause_reader.wait(METRICS_COLLECT_SECONDS):
                return

    def _current_attempt(self) -> int:
        remote = int(getattr(self._outbound, "metrics_generation", 0) or 0)
        local = int(getattr(self._outbound, "runner_generation", 1) or 1)
        return max(remote, local, 1)

    def _collect_metrics_into_pending(self) -> None:
        if self._log_dir is None:
            return
        with self._read_lock:
            points, reset = read_new_scalars(self._log_dir, self._read_state, self._current_attempt())
        if not points and not reset:
            return
        with self._lock:
            if reset:
                reset_files(self._cursor, reset)
            self._metrics_pending.extend(points)

    def _drain_metric_batches(self, max_batches: int) -> None:
        for _ in range(max_batches):
            if not self._drain_one_metric_batch():
                return

    def _drain_one_metric_batch(self) -> bool:
        with self._lock:
            if not self._metrics_pending:
                return False
            batch = self._metrics_pending[:_MAX_METRIC_BATCH]
            del self._metrics_pending[: len(batch)]
        scalars = [point for point in batch if not point.get("_advance")]
        body = [{key: value for key, value in point.items() if not str(key).startswith("_")} for point in scalars]
        if body:
            try:
                self._outbound.post_metrics(body)
            except Exception:
                with self._lock:
                    self._metrics_pending = batch + self._metrics_pending
                raise
        with self._lock:
            apply_file_ack(self._cursor, batch)
            try:
                save(self._cursor_path, self._cursor)
            except OSError as exc:
                print(f"Could not save metrics cursor ({exc}); will retry", file=sys.stderr)
        return True

    def _log_has_unread(self) -> bool:
        if self._log_dir is None:
            return False
        with self._read_lock:
            state = {key: int(entry.get("offset") or 0) for key, entry in self._read_state.items()}
        for path in iter_event_files(self._log_dir):
            rel = event_relpath(self._log_dir, path)
            if rel is None:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > state.get(rel, 0):
                return True
        return False

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

        payload: dict[str, Any] = {
            "heartbeat": heartbeat,
            "log_entries": logs,
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
            control, _metrics_accepted = self._outbound.post_sync(payload)
        except Exception:
            if logs:
                self._log_buffer = logs + self._log_buffer
            raise
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
