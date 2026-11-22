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


def _loss_point(step: int, value: float = 0.1) -> dict:
    return {
        "_event_file": "events.out.tfevents.test",
        "tag": "train/loss",
        "step": step,
        "value": value,
    }


def test_repeated_collects_queue_each_step_once(tmp_path: Path, monkeypatch) -> None:
    outbound = _FlakyOutbound()
    outbound.failures_left = 0
    coordinator = RunSyncCoordinator(outbound, cache_dir=tmp_path)
    coordinator._log_dir = tmp_path
    waves = [
        [_loss_point(1, 0.5), _loss_point(2, 0.4)],
        [_loss_point(1, 0.5), _loss_point(2, 0.3), _loss_point(3, 0.2)],
    ]

    def fake_collect(log_dir: Path, cursor: dict) -> list[dict]:
        del log_dir, cursor
        return waves.pop(0)

    monkeypatch.setattr("zust_runner.sync_coordinator.collect_new_scalars", fake_collect)
    coordinator._collect_metrics_into_pending()
    coordinator._collect_metrics_into_pending()

    assert [(point["step"], point["value"]) for point in coordinator._metrics_pending] == [
        (1, 0.5),
        (2, 0.3),
        (3, 0.2),
    ]


def test_acked_batch_discards_duplicate_steps_still_queued(tmp_path: Path) -> None:
    outbound = _FlakyOutbound()
    outbound.failures_left = 0
    coordinator = RunSyncCoordinator(outbound, cache_dir=tmp_path)
    coordinator._metrics_pending = [_loss_point(1) for _ in range(600)] + [_loss_point(2)]

    coordinator.sync_now()

    assert [point["step"] for point in coordinator._metrics_pending] == [2]
    assert coordinator._cursor["events.out.tfevents.test"]["train/loss"] == 1
    assert [point["step"] for point in outbound.payloads[0]["metric_points"]] == [1] * 500
