"""Buffer run callbacks locally when the API is unreachable, then replay in order."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import httpx

from zust_runner.http_client import is_retriable_api_error, with_transport_retries


def _is_retriable(exc: BaseException) -> bool:
    return is_retriable_api_error(exc)


class RunOutbound:
    """POST progress, logs, and metrics; persist failures under the run workspace."""

    def __init__(
        self,
        client: httpx.Client,
        *,
        api_base: str,
        run_id: str,
        headers: dict[str, str],
        cache_dir: Path,
        runner_generation: int = 1,
    ) -> None:
        self._client = client
        self._api_base = api_base.rstrip("/")
        self._run_id = run_id
        from zust_runner.http_client import merge_runner_headers

        self._headers = merge_runner_headers(headers)
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._queue_path = self._cache_dir / "pending.jsonl"
        self._lock = threading.Lock()
        self._runner_generation = runner_generation
        self._last_run_cancelled = False
        self._metrics_pending_path = self._cache_dir / "metrics-pending.jsonl"
        self._pending_control = 0
        self._hf_action_lock = threading.Lock()
        self._huggingface_action: dict[str, Any] | None = None

    def note_control(self, code: int) -> None:
        if code:
            self._pending_control = code

    def consume_control(self) -> int:
        code, self._pending_control = self._pending_control, 0
        return code

    def peek_huggingface_action(self) -> dict[str, Any] | None:
        with self._hf_action_lock:
            action = self._huggingface_action
            return dict(action) if isinstance(action, dict) else None

    def complete_runner_action(self, action: dict[str, Any]) -> None:
        from zust_runner.runner_cache import complete_action_resilient

        complete_action_resilient(
            self._client,
            api_base=self._api_base,
            run_id=self._run_id,
            headers=self._headers,
            ack_action=action,
        )

    def _remember_huggingface_action(self, pending: Any) -> None:
        action = pending if isinstance(pending, dict) and pending.get("type") == "huggingface_push" else None
        with self._hf_action_lock:
            self._huggingface_action = dict(action) if isinstance(action, dict) else None

    def flush(self) -> None:
        with self._lock:
            if not self._queue_path.exists():
                return
            raw_lines = [line for line in self._queue_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not raw_lines:
                return
            remaining: list[str] = []
            for idx, line in enumerate(raw_lines):
                record = json.loads(line)
                try:
                    self._deliver(record)
                except Exception as exc:
                    if _is_retriable(exc):
                        remaining = raw_lines[idx:]
                        break
            self._write_queue(remaining)

    def post_sync(self, payload: dict[str, Any]) -> tuple[int, int]:
        """Return (control_code, metrics_accepted)."""
        self.flush()
        try:
            return self._send_sync(payload)
        except Exception as exc:
            if _is_retriable(exc):
                self._enqueue({"type": "sync", "payload": payload})
                return 0, 0
            raise

    def post_progress(self, **fields: Any) -> int:
        """Return 0 continue, 1 cancel, 2 warm restart, 3 purge workspace."""
        control, _ = self.post_sync({"heartbeat": True, "progress": fields})
        return control

    def append_logs(self, entries: list[dict[str, str]]) -> None:
        if not entries:
            return
        control, _ = self.post_sync({"heartbeat": False, "log_entries": entries})
        if control:
            return

    def append_log(self, stream: str, line: str) -> None:
        self.append_logs([{"stream": stream, "line": line}])

    def post_workspace_snapshot(self, payload: dict[str, Any]) -> None:
        if not payload.get("entries"):
            return
        self.post_sync({"heartbeat": False, "workspace": payload})

    def post_metrics(self, points: list[dict[str, Any]]) -> None:
        if not points:
            return
        _, accepted = self.post_sync({"heartbeat": False, "metric_points": points})
        if accepted != len(points):
            self._enqueue_metrics(points[accepted:])

    def _enqueue(self, record: dict[str, Any]) -> None:
        with self._lock:
            with self._queue_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    def _write_queue(self, lines: list[str]) -> None:
        if lines:
            self._queue_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        elif self._queue_path.exists():
            self._queue_path.unlink()

    def _enqueue_metrics(self, points: list[dict[str, Any]]) -> None:
        if not points:
            return
        with self._lock:
            with self._metrics_pending_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"points": points}, separators=(",", ":")) + "\n")

    def _send_sync(self, payload: dict[str, Any]) -> tuple[int, int]:
        def _post() -> tuple[int, int]:
            response = self._client.post(
                f"{self._api_base}/api/v1/internal/runs/{self._run_id}/sync",
                headers=self._headers,
                json=payload,
                timeout=90.0,
            )
            response.raise_for_status()
            data = response.json()
            control = self._control_from_response(data)
            self.note_control(control)
            metrics_accepted = int(data.get("metrics_accepted") or 0)
            return control, metrics_accepted

        return with_transport_retries(_post)

    def _control_from_response(self, data: dict[str, Any]) -> int:
        pending = data.get("pending_action")
        self._remember_huggingface_action(pending)
        if bool(data.get("cancel")):
            self._last_run_cancelled = bool(data.get("run_cancelled"))
            return 1
        if isinstance(pending, dict) and pending.get("type") == "purge_workspace":
            return 3
        remote_generation = int(data.get("runner_generation") or self._runner_generation)
        if remote_generation > self._runner_generation:
            return 2
        return 0

    def _send_progress(self, fields: dict[str, Any]) -> int:
        def _post() -> int:
            response = self._client.post(
                f"{self._api_base}/api/v1/internal/runs/{self._run_id}/progress",
                headers=self._headers,
                json=fields,
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
            return self._control_from_response(data)

        return with_transport_retries(_post)

    def _send_logs(self, payload: dict[str, Any]) -> None:
        def _post() -> None:
            self._client.post(
                f"{self._api_base}/api/v1/internal/runs/{self._run_id}/logs",
                headers=self._headers,
                json=payload,
                timeout=60.0,
            ).raise_for_status()

        with_transport_retries(_post)

    def _send_metrics(self, payload: dict[str, Any]) -> None:
        def _post() -> None:
            self._client.post(
                f"{self._api_base}/api/v1/internal/runs/{self._run_id}/metrics",
                headers=self._headers,
                json=payload,
                timeout=60.0,
            ).raise_for_status()

        with_transport_retries(_post)

    def _send_workspace_snapshot(self, payload: dict[str, Any]) -> None:
        def _post() -> None:
            self._client.post(
                f"{self._api_base}/api/v1/internal/runs/{self._run_id}/workspace-snapshot",
                headers=self._headers,
                json=payload,
                timeout=90.0,
            ).raise_for_status()

        with_transport_retries(_post)

    def _deliver(self, record: dict[str, Any]) -> None:
        kind = record.get("type")
        payload = record.get("payload") or {}
        if kind == "sync":
            self._send_sync(payload)
        elif kind == "progress":
            self._send_progress(payload)
        elif kind == "logs":
            self._send_logs(payload)
        elif kind == "metrics":
            self._send_metrics(payload)
        elif kind == "workspace":
            self._send_workspace_snapshot(payload)
        else:
            raise ValueError(f"Unknown outbound record type: {kind!r}")
