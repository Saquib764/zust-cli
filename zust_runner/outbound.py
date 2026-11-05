"""Buffer run callbacks locally when the API is unreachable, then replay in order."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import httpx

from zust_runner.http_client import with_transport_retries

_RETRIABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


def _is_retriable(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TransportError, httpx.TimeoutException)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRIABLE_STATUS
    return False


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
        runner_epoch: int = 1,
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
        self._runner_epoch = runner_epoch

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

    def post_progress(self, **fields: Any) -> int:
        """Return 0 continue, 1 cancel, 2 warm restart, 3 purge workspace."""
        self.flush()
        try:
            return self._send_progress(fields)
        except Exception as exc:
            if _is_retriable(exc):
                self._enqueue({"type": "progress", "payload": fields})
                return 0
            raise

    def append_logs(self, entries: list[dict[str, str]]) -> None:
        if not entries:
            return
        self.flush()
        payload = {"entries": entries}
        try:
            self._send_logs(payload)
        except Exception as exc:
            if _is_retriable(exc):
                self._enqueue({"type": "logs", "payload": payload})
                return
            raise

    def append_log(self, stream: str, line: str) -> None:
        self.append_logs([{"stream": stream, "line": line}])

    def post_workspace_snapshot(self, payload: dict[str, Any]) -> None:
        if not payload.get("entries"):
            return
        self.flush()
        try:
            self._send_workspace_snapshot(payload)
        except Exception as exc:
            if _is_retriable(exc):
                self._enqueue({"type": "workspace", "payload": payload})
                return
            raise

    def post_metrics(self, points: list[dict[str, Any]]) -> None:
        if not points:
            return
        self.flush()
        payload = {"points": points}
        try:
            self._send_metrics(payload)
        except Exception as exc:
            if _is_retriable(exc):
                self._enqueue({"type": "metrics", "payload": payload})
                return
            raise

    def _enqueue(self, record: dict[str, Any]) -> None:
        with self._lock:
            with self._queue_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    def _write_queue(self, lines: list[str]) -> None:
        if lines:
            self._queue_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        elif self._queue_path.exists():
            self._queue_path.unlink()

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
            if bool(data.get("cancel")):
                return 1
            pending = data.get("pending_action")
            if isinstance(pending, dict) and pending.get("type") == "purge_workspace":
                return 3
            remote_epoch = int(data.get("runner_epoch") or self._runner_epoch)
            if remote_epoch > self._runner_epoch:
                return 2
            return 0

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
        if kind == "progress":
            self._send_progress(payload)
        elif kind == "logs":
            self._send_logs(payload)
        elif kind == "metrics":
            self._send_metrics(payload)
        elif kind == "workspace":
            self._send_workspace_snapshot(payload)
        else:
            raise ValueError(f"Unknown outbound record type: {kind!r}")
