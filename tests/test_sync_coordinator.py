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

    def post_metrics(self, points: list[dict]) -> dict:
        self.payloads.append({"metric_points": points})
        return {"accepted": len(points), "rejected": [], "metrics_generation": 1}

    def peek_huggingface_action(self):
        return None

    runner_generation = 1
    metrics_generation = 1


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


def _loss_point(step: int, value: float = 0.1, offset: int = 0) -> dict:
    return {
        "_file": "events.out.tfevents.test",
        "_offset": offset or step,
        "_dev": 1,
        "_ino": 1,
        "tag": "train/loss",
        "step": step,
        "value": value,
        "attempt": 1,
    }


def test_repeated_reads_append_new_records(tmp_path: Path, monkeypatch) -> None:
    outbound = _FlakyOutbound()
    outbound.failures_left = 0
    coordinator = RunSyncCoordinator(outbound, cache_dir=tmp_path)
    coordinator._log_dir = tmp_path
    waves = [
        ([_loss_point(1, 0.5, 10), _loss_point(2, 0.4, 20)], []),
        ([_loss_point(3, 0.2, 30)], []),
    ]

    def fake_read(log_dir: Path, state: dict, attempt: int):
        del log_dir, state, attempt
        return waves.pop(0)

    monkeypatch.setattr("zust_runner.sync_coordinator.read_new_scalars", fake_read)
    coordinator._collect_metrics_into_pending()
    coordinator._collect_metrics_into_pending()

    assert [(point["step"], point["value"]) for point in coordinator._metrics_pending] == [
        (1, 0.5),
        (2, 0.4),
        (3, 0.2),
    ]


def test_acked_batch_advances_file_offset(tmp_path: Path) -> None:
    outbound = _FlakyOutbound()
    outbound.failures_left = 0
    coordinator = RunSyncCoordinator(outbound, cache_dir=tmp_path)
    coordinator._metrics_pending = [_loss_point(1, offset=index) for index in range(1, 601)] + [
        _loss_point(2, offset=601)
    ]

    coordinator.sync_now()

    assert coordinator._metrics_pending[-1]["step"] == 2
    assert len(coordinator._metrics_pending) == 101
    assert coordinator._cursor["files"]["events.out.tfevents.test"]["offset"] == 500
    metric_payloads = [item for item in outbound.payloads if "metric_points" in item]
    assert len(metric_payloads[0]["metric_points"]) == 500
