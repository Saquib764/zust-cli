"""Reporter keeps running when the API disappears."""

import time
from pathlib import Path

from zust_runner.sync_coordinator import RunSyncCoordinator


class _FlakyOutbound:
    def __init__(self) -> None:
        self.payloads: list[dict] = []
        self.failures_left = 1

    def post_sync(self, payload: dict) -> tuple[int, int]:
        if self.failures_left:
            self.failures_left -= 1
            raise RuntimeError("api down")
        self.payloads.append(payload)
        return 0, len(payload.get("metric_points") or [])

    def peek_huggingface_action(self):
        return None


def test_failed_post_puts_logs_back(tmp_path: Path) -> None:
    outbound = _FlakyOutbound()
    coordinator = RunSyncCoordinator(outbound, cache_dir=tmp_path)
    coordinator.buffer_log("stderr", "step 10984")

    try:
        coordinator.sync_now(heartbeat=True)
        raised = False
    except RuntimeError:
        raised = True
    assert raised is True
    assert outbound.payloads == []

    coordinator.sync_now(heartbeat=True)
    assert outbound.payloads[0]["log_entries"] == [{"stream": "stderr", "line": "step 10984"}]


def test_sync_loop_retries_after_a_failed_post(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("zust_runner.sync_coordinator.SYNC_INTERVAL_SECONDS", 0.01)
    outbound = _FlakyOutbound()
    outbound.failures_left = 2
    coordinator = RunSyncCoordinator(outbound, cache_dir=tmp_path)
    coordinator.buffer_log("stderr", "still training")
    coordinator.start()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not outbound.payloads:
        time.sleep(0.02)
    coordinator.stop()
    assert outbound.payloads
    assert outbound.payloads[0]["log_entries"] == [{"stream": "stderr", "line": "still training"}]
