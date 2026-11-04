"""Outbound cache when the API is temporarily unavailable."""

import json
from pathlib import Path

import httpx

from zust_runner.outbound import RunOutbound


def test_queues_logs_on_503_then_flushes(tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/logs"):
            calls.append(("logs", request.content.decode()))
            if len(calls) == 1:
                return httpx.Response(503)
            return httpx.Response(200, json={"data": None})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="https://api.test")
    outbound = RunOutbound(
        client,
        api_base="https://api.test",
        run_id="run_1",
        headers={"Authorization": "Bearer t"},
        cache_dir=tmp_path,
    )

    outbound.append_log("stdout", "hello")
    assert outbound._queue_path.exists()
    queued = json.loads(outbound._queue_path.read_text().strip())
    assert queued["type"] == "logs"

    outbound.flush()
    assert not outbound._queue_path.exists()
    assert len(calls) == 2


def test_progress_does_not_raise_on_transport_error(tmp_path: Path) -> None:
    transport = httpx.MockTransport(lambda _request: httpx.Response(503))
    client = httpx.Client(transport=transport, base_url="https://api.test")
    outbound = RunOutbound(
        client,
        api_base="https://api.test",
        run_id="run_1",
        headers={"Authorization": "Bearer t"},
        cache_dir=tmp_path,
    )
    assert outbound.post_progress(phase="training") is False
    assert outbound._queue_path.exists()
