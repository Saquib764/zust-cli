"""Shared httpx client for runner → API callbacks."""

from __future__ import annotations

import os
import time
from typing import Callable, TypeVar

import httpx

_T = TypeVar("_T")

_DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=45.0)


def api_base() -> str:
    return os.environ.get("ZUST_API_BASE", "").rstrip("/")


def merge_runner_headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    headers = dict(extra or {})
    if "ngrok" in api_base().lower():
        headers.setdefault("ngrok-skip-browser-warning", "69420")
    return headers


def create_runner_client() -> httpx.Client:
    return httpx.Client(
        timeout=_DEFAULT_TIMEOUT,
        follow_redirects=True,
        headers=merge_runner_headers(),
    )


def with_transport_retries(action: Callable[[], _T], *, attempts: int = 6) -> _T:
    delay = 1.0
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return action()
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            last_error = exc
            if attempt >= attempts - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 1.5, 10.0)
    if last_error is not None:
        raise last_error
    raise RuntimeError("retry loop exited unexpectedly")
